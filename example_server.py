import numpy as np
import cv2

from time import time

from urllib.request import urlopen

# times = {"start": time()}

def main():
    url = input("Remote server URL: ")
    try:
        st = time()
        frames = 0
        while True:
            opener = urlopen(url)
            npimage = cv2.imdecode(np.frombuffer(opener.read(), np.uint8), -1)
            cv2.imshow('image',npimage)
            frames += 1
            endt = time()
            if endt - st >= 1:
                print("              ", end="\r")
                print(f"FPS: {frames}", end="\r")
                frames = 0
                st = endt
            cv2.waitKey(1)
            # obj = BytesIO(opener.read())
    except KeyboardInterrupt:
        return 0

def main2():
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/', methods=["POST"])
    def receive():
        nparr = np.fromstring(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(f"Image received - size: {img.shape[1]}x{img.shape[0]}")
        return jsonify({"status": 200}), 200

    app.run(port=5000)


main2()

