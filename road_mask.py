import cv2
from matplotlib import pyplot as plt

import numpy as np
from skimage.draw import polygon2mask

from urllib.request import urlopen
from time import time

from flask import Flask
from flask import request
from flask import jsonify

from led_control import led_control


class CarDetector:
    def __init__(self, xml_file: str = "cars.xml"):
        self._cascade = cv2.CascadeClassifier(xml_file)

    @staticmethod
    def _masked_image(image, polygons, show_result=False):
        mask = sum([polygon2mask(image.shape[:2], polygon)
                    for polygon in polygons])
        image[:, :, 0] = image[:, :, 0] * mask
        image[:, :, 1] = image[:, :, 1] * mask
        image[:, :, 2] = image[:, :, 2] * mask

        if show_result is True:
            plt.figure(figsize=(40, 40))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('masked image result')
            plt.show()

        return image

    def _detection(self, img, show_result=False):
        cars = self._cascade.detectMultiScale(img, 1.1, 1)
        if show_result:
            for (x, y, w, h) in cars:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            plt.figure(figsize=(40, 40))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('masked and car detecetion image result')
            plt.show()

        return cars

    def detect(self, image, show_result=False):
        img = image[150:, :]
        h, w = img.shape[:2]

        first_polygon = np.array([[0, 100],
                                  [h / 2, 60],
                                  [4 * h / 5, 0],
                                  [h, 0],
                                  [h, 60],
                                  [h / 2, 120],
                                  [0, 132]])
        second_polygon = np.array([[0, 135],
                                   [h / 2, 125],
                                   [h, 105],
                                   [h, 205],
                                   [0, 157]])
        third_polygon = np.array([[0, 185],
                                  [h, 325],
                                  [h, 440],
                                  [0, 220]])

        polygons = [first_polygon, second_polygon, third_polygon]

        return [self._detection(self._masked_image(img.copy(),
                                                   [polygon]),
                                show_result=show_result)
                for _, polygon in enumerate(polygons)]


app = Flask(__name__)
detector = CarDetector()


@app.route('/', methods=["POST"])
def receive_image():
    nparr = np.frombuffer(request.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    vehicles = [len(x) for x in detector.detect(image)]
    return jsonify({"vehicles": vehicles}), 200


@app.route("/data", methods=["POST"])
def receive_semaphore_status():
    status = request.get_json()
    if status is None:
        return jsonify({"status": "error"}), 400
    mode = status.get("mode")
    if mode is None:
        return "No mode", 400
    led_control("{0:b}".format(mode))
    return "Ok", 200


app.run(port=5000)


# pimg = "885.jpg"
# oimage = cv2.imread(pimg)
# detector = CarDetector()
# rs = detector.detect(oimage, show_result=False)
# for i in rs:
#     print(len(i))
# print(rs)

# main()
