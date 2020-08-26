import cv2
import numpy as np

path = "/home/okamoto/kurodake/images/010625.JPG"

def snow_watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    edge = 255 - edge
    gray[edge==0]=0
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    sure_snow = cv2.erode(opening, kernel, iterations=3)
    # Finding unknown region
    sure_snow = np.uint8(sure_snow)
    unknown = cv2.subtract(sure_bg, sure_snow)
    snow = sure_snow + 1
    snow[unknown==255]=0
    snow[sure_snow==255]=2
    snow = np.int32(snow)
    snow = cv2.watershed(img,snow)
    snow = np.uint8(snow)
    snow[snow==2]=255
    snow[snow!=255]=0
    return snow

def snow_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh


"""
# example
import glob
import os

d = "/home/okamoto/kurodake/images/"
out_d = "/home/okamoto/kurodake/watershed/"
os.makedirs(out_d)
files = glob.glob("/home/okamoto/kurodake/images/*")

for f in files:
    img = cv2.imread(f)
    snow = snow_watershed(img)
    out_path = f.replace(d, out_d)
    cv2.imwrite(out_path, snow)
"""
