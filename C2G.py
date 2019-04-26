import cv2
import numpy as np
import csv

def getGreyscale(image):
    grey = [ [ 0 for i in range(len(image[0]))] for j in range(len(image))]
    for i in range(len(grey)):
        for j in range(len(grey[0])):
            val = (image[i][j][0]  *.07 + image[i][j][1]*.72 + image[i][j][2] *.21)
            if val > 255:
                grey[i][j] = 255
            elif val < 0:
                grey[i][j] = 0
            else:
                grey[i][j] = val
    grey = np.array(grey, dtype=np.uint8)
    return grey

def showImg(img):
    cv2.imshow('Display', img)
    cv2.waitKey()

img = cv2.imread('flowers.jpg', 1)
showImg(img)
greyscale = getGreyscale(img)
showImg(greyscale)

values = []

for i in range((len(img))/8):
    for j in range((len(img[0]))/4):
        values.append([img[i][j][0], img[i][j][1], img[i][j][2], greyscale[i][j]])

with open('pixelVals4.csv', 'wb') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(values)

csvFile.close()
