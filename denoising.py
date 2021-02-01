# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:58:16 2021

@author: Owner
"""

import cv2, sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from matplotlib import pyplot as plt


imageFile='wafermapdefects.jpg'
img = cv2.imread(imageFile, cv2.IMREAD_COLOR)



blur = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(blur, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)



contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

contours_xy = np.array(contours)
contours_xy.shape

x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
print(x_min)
print(x_max)
 
# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
print(y_min)
print(y_max)



# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min
img_trim = img[y:y+h, x:x+w]
cv2.imwrite('org_trim.jpg', img_trim)
trim = cv2.imread('org_trim.jpg')


denoised_img = cv2.fastNlMeansDenoisingColored(trim, None, 17, 15, 7, 21)
 

cv2.imshow("after", denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''  이건 https://red-pulse.tistory.com/190 참고!!
kernel = np.ones((5,5),np.uint8)
# para1 : 이미지, para2 : 커널, para3 : erode 반복 횟수
erode = cv2.erode(trim,kernel,iterations = 1)
# para1 : 이미지, para2 : 커널, para3 : dilate 반복 횟수
dilate = cv2.dilate(trim,kernel,iterations = 1)

#결과 보기
#cv2.imshow("original",img)
#cv2.imshow("erode",erode)
#cv2.imshow("dilate",dilate)
'''

# para1 : 이미지, para2 : 함수 이용, para3 : 커널
gradient = cv2.morphologyEx(denoised_img, cv2.MORPH_GRADIENT, kernel)

#gradient결과 보기

cv2.imshow("gradient",gradient)

cv2.waitKey(0)

closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

#closing결과 보기 => 이건 디노이즈 + gradient + closing 한거임!

cv2.imshow("closing.jpg",closing)

cv2.waitKey(0)



tophat = cv2.morphologyEx(denoised_img, cv2.MORPH_TOPHAT, kernel)

#결과 보기. 이거는 디노이즈에 tophat

cv2.imshow("tophat.jpg",tophat)

cv2.waitKey(0)




gray_for_wafer = cv2.cvtColor(trim, cv2.COLOR_BGR2GRAY)

gray_for_defect = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

#gray = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)


plt.imshow(gray_for_defect, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(gray_for_defect)
print(gray_for_defect)

'''
이거 대비높히는건데 잘 안되서 그냥 다 지울지 고민중인 부분임!!
histogram, bin = np.histogram(img.ravel(), 256, [0, 256]) 
cumsum = histogram.cumsum() 
LUT = np.uint8((cumsum - cumsum.min()) * 255 / (cumsum.max() - cumsum.min())) 
equ = LUT[gray] 
hist = cv2.equalizeHist(gray)

cv2.imshow("original", gray) 
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imshow('result1', equ) 
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imshow('result2', hist)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''



# 이거는 웨이퍼 동그라미만 검출하는거!
ret, wafer = cv2.threshold(gray_for_wafer, 254, 1, cv2.THRESH_BINARY_INV)

img_height, img_width = wafer.shape

wafer_26 = np.zeros( (26,26), dtype = wafer.dtype ) 

new_height = img_height//26 
new_width = img_width//26  

for j in range(26):
    for i in range(26):
        y = j*new_height
        x = i*new_width  
        pixel = wafer[y:y+new_height, x:x+new_width]  
        wafer_26[j,i] = pixel.sum(dtype='int64')//(new_height*new_width) 
plt.imshow(wafer_26, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(wafer_26)
print(wafer_26)


#디펙을 한번 검출하면 자연스럽게 웨이퍼 가장자리도 디펙으로 읽힘 그래서 디펙쪽은 threshold를 두번 주는거임!!
ret, defect = cv2.threshold(gray_for_defect, 20, 255, cv2.THRESH_BINARY_INV)


img_height, img_width = defect.shape

defect_26 = np.zeros( (26,26), dtype = defect.dtype ) 

new_height = img_height//26 
new_width = img_width//26  

for j in range(26):
    for i in range(26):
        y = j*new_height
        x = i*new_width  
        pixel = defect[y:y+new_height, x:x+new_width]  
        defect_26[j,i] = pixel.sum(dtype='int64')//(new_height*new_width) 

plt.imshow(defect_26, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(defect_26)
print(defect_26)

''' 디펙 두번 하는데 문턱값이 잘 안되서 조정중이어서 잠시 주석처리

ret, defect2 = cv2.threshold(defect, 1, 4, cv2.THRESH_BINARY)


img_height, img_width = defect2.shape

defect2 = np.zeros( (26,26), dtype = defect2.dtype ) 

new_height = img_height//26 
new_width = img_width//26  

for j in range(26):
    for i in range(26):
        y = j*new_height
        x = i*new_width  
        pixel = defect2[y:y+new_height, x:x+new_width]  
        defect2[j,i] = pixel.sum(dtype='int64')//(new_height*new_width) 


plt.imshow(defect2, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(defect2)
print(defect2)

'''

'''
newImage= defect2 + wafer

plt.imshow(newImage, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(newImage)
print(newImage)
'''


'''
img_height, img_width = kk.shape

newImage = np.zeros( (26,26), dtype = kk.dtype ) 

new_height = img_height//26 
new_width = img_width//26  

for j in range(26):
    for i in range(26):
        y = j*new_height
        x = i*new_width  
        pixel = kk[y:y+new_height, x:x+new_width]  
        newImage[j,i] = pixel.sum(dtype='int64')//(new_height*new_width) 


plt.imshow(newImage, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(newImage)
print(newImage)
'''