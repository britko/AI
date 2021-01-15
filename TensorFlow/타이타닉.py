import os
import sys

import numpy as np      #수치처리 모듈
import pandas as pd     #데이터처리 모듈
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# 데이터셋 불러오기.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')  #pop: 해당 데이터를 지운다
y_eval = dfeval.pop('survived')

print(dftrain.shape)    #(627, 9)
print(dftrain.head())   #head: 행렬의 앞 부분을 출력

dftrain.age.hist(bins=20)
plt.show()