import matplotlib.pyplot as plt
from matplotlib.image import imread

 #이미지 읽어오기(이미지 경로 설정: 앞에 붙는 r은 \U가 이스케이프 문자로 읽히는걸 방지하는 것)
img = imread(r'C:\Users\pc\Desktop\새 폴더\디퓨져 스티커_20191210155832_melisse.jpg')

plt.imshow(img)
plt.show()