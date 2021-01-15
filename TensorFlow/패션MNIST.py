# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    #훈련 데이터, 테스트 데이터를 튜플로 저장

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   #분류할 이름을 0~9번까지 배열로 저장

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

plt.figure()
plt.imshow(train_images[0]) #이미지 출력
plt.colorbar()
plt.grid(False) #격자 표시 안함 plt.grid
plt.show()

#신경망 모델에 주입하기 전에 픽셀 값의 범위를 0~1사이로 조정
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10)) #figure생성(figsize: figure 크기 조절 n*n)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])  #x축 눈금 표시
    plt.yticks([])  #y축 눈금 표시
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([  #레이어를 선형으로 연결
    keras.layers.Flatten(input_shape=(28, 28)),     #2차원 배열(28*28픽셀)의 이미지 포맷을 784픽셀의 1차원 배열로 변환
    keras.layers.Dense(128, activation='relu'),     #[ ,128]형태의 배열(노드)을 출력 / 활성화 함수: relu
    keras.layers.Dense(10, activation='softmax')    #10의 확률을 반환하고 반환된 값의 전체 합은 1이다. 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력
])

#모델 컴파일
model.compile(optimizer='adam',     #옵티마이저 - 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정
              loss='sparse_categorical_crossentropy',   #손실 함수 - 훈련하는 동안 모델의 오차를 측정
              metrics=['accuracy']) #지표 - 훈련 단계와 테스트 단계를 모니터링하기 위해 사용

#모델 훈련
model.fit(train_images, train_labels, epochs=5)

#정확도 평가 > 0.87(훈련 데이터가 오버피팅돼서 테스트에서 낮은 정확도를 보임)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

#예측 만들기 - 훈련된 모델을 사용하여 이미지에 대한 예측을 만든다.
predictions = model.predict(test_images)    #테스트 세트에 있는 각 이미지의 레이블을 예측
predictions[0]  #10개의 옷 품목에 상응하는 모델의 신뢰도를 나타냄
np.argmax(predictions[0])   #가장 높은 신뢰도를 가진 레이블 찾기(class_name[9])
test_labels[0]  #테스트 세트의 0번이 class_name[9]가 맞는지 확인

#10개의 클래스에 대한 예측을 그래프로 표현
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[0]

print(img.shape)

# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])