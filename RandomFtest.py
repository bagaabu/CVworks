# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:48:54 2016

@author: abulin
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import sys
import operator
from collections import Counter
from skimage.feature import hog
from sklearn.externals import joblib


############################## testing part  #########################
clf = joblib.load('bag0.pkl')
im = cv2.imread('legosma30.tiff')
im = cv2.resize(im, (0,0), fx=1.5, fy=1.5)
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
point = []
positiony = []
positionh = []
toleranty = 1
toleranth = 1
T = 0

while (1):
    T = 0
    positiony = []
    positionh = []
    for cnt in contours:
        if cv2.contourArea(cnt)>15:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>=10:
                point = [x,y,w,h]
                point = list(point)
                point[1] = point[1]/toleranty
                point[3] = point[3]/toleranth
                positiony.append(point[1])
                positionh.append(point[3])
    Couy = Counter(positiony)
    Mosty = Couy.most_common(1)[0][0]
    Numy = Couy.most_common(1)[0][1]
    Couh = Counter(positionh)
    Mosth = Couh.most_common(1)[0][0]
    Numh = Couh.most_common(1)[0][1]
    if toleranty >=7 or toleranth>=7:
        break
    if Numh == Numy == 5:
        break
    else:
        if Numy != 5:
            toleranty = toleranty+1
        if Numh != 5:
            toleranth = toleranth+1


for cnt in contours:
    more = []
    if cv2.contourArea(cnt)>15:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>=10:
            if y/toleranty == Mosty:
                if h/toleranth == Mosth:
                    T=T+1 
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roi_canny = cv2.Canny(roi,50,125)
                    roismall = cv2.resize(roi,(16,16))
                    fd, hog_image = hog(roismall, orientations=16, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)               
                    roismall_hog = fd.reshape((1,fd.size))             
                    roismall_canny = cv2.resize(roi_canny,(16,16))
                    roismall = roismall.reshape((1,256))
                    roismall_canny = roismall_canny.reshape((1,256))
                    roismall = np.float32(roismall)
                    roismall_canny = np.float32(roismall_canny)
                    results = clf.predict(roismall)
                    results_canny = clf.predict(roismall_canny)
                    results_hog = clf.predict(roismall_hog)
                    if  results == results_canny == results_hog:
                        string = str(int((results)))
                        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
                    else: 
                        more = np.append(more,results)
                        more = np.append(more,results_canny)
                        more = np.append(more,results_hog)
                        string = str(int((Counter(more).most_common(1)[0][0])))
                        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)
cv2.destroyAllWindows()