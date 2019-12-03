import cv2
from matplotlib import pyplot as plt
from skimage import data
from skimage.filters import threshold_local
import glob
import numpy as np
from skimage.draw import polygon, polygon2mask
from skimage.color import rgb2gray
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from urllib.request import urlopen
from time import time



def masked_image(img, polygons, show_result=False):
    """Create a mask based on polygons.
    
    Arguments:
    image -- the image for which the mask will be used
    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    mask = sum([polygon2mask(img.shape[:2],polygon)for polygon in polygons])
    img[:, :, 0] = img[:, :, 0] * mask
    img[:, :, 1] = img[:, :, 1] * mask
    img[:, :, 2] = img[:, :, 2] * mask

    if show_result is True:
        
        plt.figure(figsize = (40,40))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('masked image result')
        plt.show()
    
    return img





def car_detection(img, show_result=False):
    cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)

    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    if show_result is True:
        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)      


        plt.figure(figsize = (40,40))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('masked and car detecetion image result')
        plt.show()
        
    return cars





def car_detector(img):
    
    img = img[150:,:]
    
    heigth, width = img.shape[:2]
    

    
    first_polygon = np.array([[0, 100], [heigth / 2, 60], [4 * heigth / 5, 0], [heigth, 0],[heigth, 60],[heigth / 2, 120], [0, 132]])
    second_polygon = np.array([[0, 135], [heigth / 2, 125], [heigth, 105], [heigth, 205], [0, 157]])
    thrid_polygon = np.array([[0, 185], [heigth, 325],[heigth,440], [0,220]])  
    
    
    polygons=[first_polygon, second_polygon, thrid_polygon]
    
    return [car_detection(masked_image(img.copy(), [polygon])) for i, polygon in enumerate(polygons)]



def main(url=None,debug=False):

    if url is not None:
        
        url = input("Remote server URL: ")
        
    try:
        st = time()
        frames = 0
        while True:
            opener = urlopen(url)
            npimage = cv2.imdecode(np.frombuffer(opener.read(), np.uint8), -1)
            car_detector(npimage)
            if debug is True:
                frames += 1
                endt = time()
                if endt - st >= 1:
                    print("              ", end="\r")
                    print(f"FPS: {frames}", end="\r")
                    frames = 0
                    st = endt
    

main()












