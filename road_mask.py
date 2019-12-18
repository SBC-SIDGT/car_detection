import cv2
import numpy as np
import logging

from logging.handlers import RotatingFileHandler
from matplotlib import pyplot as plt
from skimage.draw import polygon2mask

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
handler = RotatingFileHandler("/var/log/roadmask.log",
                              maxBytes=10000,
                              backupCount=5)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)


@app.route('/', methods=["POST"])
def receive_image():
    nparr = np.frombuffer(request.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    app.logger.info(f"Received image whose size is: {image.shape[1]}x"
                    f"{image.shape[0]}")
    vehicles = [len(x) for x in detector.detect(image)]
    app.logger.info(f"Detected vehicles: {vehicles}")
    return jsonify({"vehicles": vehicles}), 200


@app.route("/data", methods=["POST"])
def receive_semaphore_status():
    status = request.get_json()
    if status is None:
        app.logger.error("Received no json")
        return jsonify({"status": "error"}), 400
    mode = status.get("mode")
    if mode is None:
        app.logger.error("No mode in received json")
        return "No mode", 400
    app.logger.info(f"Received json: {mode}")
    led_control(f"{mode:03b}")
    return "Ok", 200


app.run(port=5000, threaded=True, processes=4)
