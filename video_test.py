import cv2
from matplotlib import pyplot as plt
from skimage import data
from skimage.filters import threshold_local
import glob
import numpy as np
from skimage.draw import polygon, polygon2mask
# from skimage.color import rgb2gray
from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import time

def masked_image(img, polygons, show_result=False):
    """Create a mask based on polygons.

    Arguments:
    image -- the image for which the mask will be used
    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    mask = sum([polygon2mask(img.shape[:2], polygon) for polygon in polygons])
    img[:, :, 0] = img[:, :, 0] * mask
    img[:, :, 1] = img[:, :, 1] * mask
    img[:, :, 2] = img[:, :, 2] * mask

    if show_result is True:
        plt.figure(figsize=(40, 40))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('masked image result')
        plt.show()

    return img


def car_detection(img, show_result=True):
    cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)

    cars = car_cascade.detectMultiScale(img, 1.1, 1, minSize=(75, 75))
    if show_result is True:
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 400)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, f"Numero de coches: {len(cars)}",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)



    return img


def main():
    import numpy as np
    import cv2

    cap = cv2.VideoCapture('t_video5841273973463058335.mp4')

    ret, frame = cap.read()
    frame = frame[150:, :]

    heigth, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, ( width, heigth))

    while (True):
        start_time = time.time()  # start time of the loop


        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = frame[150:,:]

        heigth, width = frame.shape[:2]

        first_polygon = np.array([[0, 180], [0, 165], [250, 0], [250, 250]])
        #second_polygon = np.array([[0, 210], [0, 160], [150, 255], [100, width]])

        polygons = [first_polygon]

        # Display the resulting frame
        #cv2.imshow('frame', car_detection(masked_image(frame.copy(), [polygons])))

        img = car_detection(masked_image(frame.copy(), [polygons]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 300)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, f"FPS:  {1.0 / (time.time() - start_time)}",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        print()

        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()

    cv2.destroyAllWindows()
    for x in range(2400):
        _, frame = cap.read()
    img = frame

    plt.figure(figsize=(40, 40))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('masked and car detecetion image result')
    plt.show()

    return [car_detection(masked_image(img.copy(), [polygon])) for i, polygon in enumerate(polygons)]
    # x,y,w,h = cars[0]
    # print(Point(y + h / 2,x + w / 2))
    # print([first_polygon.contains(Point(x + w / 2, y + h / 2)) for x,y,w,h in cars])

main()