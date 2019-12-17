import cv2
from matplotlib import pyplot as plt

import numpy as np
from skimage.draw import polygon2mask

from urllib.request import urlopen
from time import time

from flask import Flask
from flask import request
from flask import jsonify


class CarDetector:
    def __init__(self, xml_file: str = "cars.xml"):
        self._cascade = cv2.CascadeClassifier(xml_file)

    @staticmethod
    def _masked_image(image, polygons, show_result=False):
        mask = sum([polygon2mask(image.shape[:2], polygon)
                    for polygon in polygons])
        img[:, :, 0] = img[:, :, 0] * mask
        img[:, :, 1] = img[:, :, 1] * mask
        img[:, :, 2] = img[:, :, 2] * mask

        if show_result is True:
            plt.figure(figsize=(40, 40))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('masked image result')
            plt.show()

        return img

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


# def main(url=None, debug=False):
#     # if url is not None:
#     # url = input("Remote server URL: ")
#
#     try:
#         st = time()
#         frames = 0
#         while True:
#             # opener = urlopen(url)
#             npimage = cv2.imdecode(np.frombuffer(opener.read(), np.uint8), -1)
#             car_detector(npimage)
#             if debug is True:
#                 frames += 1
#                 endt = time()
#                 if endt - st >= 1:
#                     print("              ", end="\r")
#                     print(f"FPS: {frames}", end="\r")
#                     frames = 0
#                     st = endt
#     except KeyboardInterrupt:
#         exit(0)
#
#
# app = Flask(__name__)
#
#
# @app.route('/', methods=["POST"])
# def receive_image():
#     nparr = np.frombuffer(request.data, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return jsonify({"status": 200}), 200

img = "885.jpg"
image = cv2.imread(img)
detector = CarDetector()
rs = detector.detect(image, show_result=False)
for i in rs:
    print(len(i))
print(rs)

# main()
