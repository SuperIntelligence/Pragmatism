# 학습일정
## [1일차](./Day%201/Day%201.md)
* 환경설정
* Numpy, Matplotlib, Pandas
## [2일차](./Day%202/Day%202.md)
* MCP 뉴런
* 퍼셉트론
* 신경망
  - 다차원 배열 계산
  - 신경망 구현
  - 출력층
  - 신경망 적용
  - 손실 함수
  - 최적화
## [3일차](./Day%203/Day%203.md)
* 신경망 학습
  - 오차 역전파
  - 계층 구현
  - 활성화 함수 계층 구현
  - Affine/Softmax 계층 구현
  - 오차역전파 구현
  - 매개변수 상신
## [4일차](./Day%204/Day%204.md)
* Tensorflow
* Keras
* 합성곱 신경망(CNN)
  - 구현과 활용
## [5일차](./Day%205/Day%205.md)
* Keras 고급 활용
* 재귀 신경망(RNN)
  - 구현과 활용
 
# 교재
밑바닥부터 시작하는 딥러닝 (http://www.yes24.com/24/goods/35519439)
 
# 요약
## 1일차
### 환경구성
anaconda 설치하셈. (https://anaconda.org/anaconda)
### 파이썬 기본
자세한건 [ML 요약 해당부분](./../1.%20ML/0%20ML%20요약.md) 참조. <p />
일반적으로 데이터는 다음과 같이 처리한다. (train_test_split 설명도 ML 요약 참조)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline

data = pd.read_csv('data/wdbc.data')
print(data.head(5), data.shape) # 한번 확인

y = data['class']
X = data.values[:,1:].astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

## 2일차
### MCP 뉴런
```python
def 뉴런(x1, x2, w1, w2, b):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```
AND, NAND, OR 게이트는 구조가 같으며 단순히 가중치(weight)와 편향(b) 값만 조정해서 구현할 수 있다.
#### AND 게이트
```python
def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    return 뉴런(x1, x2, w1, w2, b)
```
#### NAND 게이트
```python
def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    return 뉴런(x1, x2, w1, w2, b)
```
#### OR 게이트
```python
def OR(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.2
    return 뉴런(x1, x2, w1, w2, b)
```
### 퍼셉트론
퍼셉트론은 기존의 뉴런 개념에 오차에 대한 피드백 개념이 추가된 것이다. 
이를 통해 가중치와 편향을 스스로 학습하는것. 아래 코드는 결국 위의 뉴런에서 fit역할을 하는 퍼셉트론 class가 추가된 것이다.
```python
class 뉴런:
    def net_input(self, X):
        z = np.dot(X, self.w) + self.b
        return z
    
    def predict(self, X):
        z = self.net_input(X)
        y = np.where(z > 0, 1, -1)
        return y
    
class 퍼셉트론(뉴런):
    def __init__(self, 학습률, 학습횟수):
        self.학습률 = 학습률
        self.학습횟수 = 학습횟수
        
    def fit(self, X, y):
        # 가중치 초기화
        self.w = np.zeros(X.shape[1])
        self.b = 0.
        
        # 훈련 
        error_history = []
        for i in range(self.학습횟수):
            # 각 샘플별
            오류제곱합 = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                오류제곱합 += error ** 2
                update = error * self.학습률
                self.w += update * xi
                self.b += update
            error_history.append(오류제곱합)
        return error_history
```
이걸 이용해서 iris data를 실제로 학습해 보겠다.
```python
iris = pd.read_csv('data/iris.data', header=None)

y = iris[4]
X = iris.values[:, 0:4].astype('float32')

X1 = X[:100]
y1 = y[:100]    # y결과가 두종류만 나오게 하기위해 앞에서 100개만 쓴다(원래 세종류).

y1 = np.where(y1 == 'Iris-versicolor', 1, -1)   # 1, -1로 encoding

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)

model = 퍼셉트론(학습률=0.01, 학습횟수=10)
error_history = model.fit(X, y)     # 학습!
plt.plot(error_history, 'go--')     # scikit-learn엔 없지만 학습 잘되는지 보려고 만듬.
```
### 신경망 
#### 활성화함수
```python
def step(x):
    return np.where(x>0, 1, -1)

x = np.arange(-5., 5., 0.1)

plt.plot(x, step(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

plt.plot(x, sigmoid(x))

def relu(x):
    return np.maximum(0, x)

plt.plot(x, relu(x))
```
#### 소프트맥스
```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
```
#### 신경망구현
```python
class Layer:
    def __init__(self, 입력수, 출력수, 활성화_함수):
        self.W = np.random.randn(입력수, 출력수)
        self.b = np.random.randn(출력수)
        self.activation = 활성화_함수
        
    def output(self, X):
        z = np.dot(X, self.W) + self.b
        y = self.activation(z)
        return y
```
레이어를 만들었다. 순전파를 만들어보자
```python
class FeedForwardNet:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        """순전파"""
        layer_input = X
        for layer in self.layers:
            layer_input = layer.output(layer_input)
        
        y = layer_input
        return y
```
이제 이런식으로 MNIST 훈련 데이터에 맞는 신경망을 구성할 수 있다.
```python
layer1 = Layer(입력수=784, 출력수=50, 활성화_함수=sigmoid)
layer2 = Layer(50, 100, sigmoid)
layer3 = Layer(100, 10, softmax)
```
우선은 훈련된 가중치를 가져와 사용해보자
```python
import pickle

with open('data/mnist_weight.pkl', 'rb') as fp:
    params = pickle.load(fp)

layer1.W = params['W1']
layer1.b = params['b1']
layer2.W = params['W2']
layer2.b = params['b2']
layer3.W = params['W3']
layer3.b = params['b3']
```
그럼 한번 순전파해서 잘되나 검증을 해보자.
```python
model = FeedForwardNet()

for layer in [layer1, layer2, layer3]:
    model.add(layer)

y_pred = np.argmax(model.predict(X_test), axis=1)
np.mean(y_pred == y_test)
```
우린 FeedFowardNet class에 fit과 copute_loss도 만들어야함
```python
    def compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_func(y_pred, y)
    
    def fit(self, X, Y, batch_size):
        loss_history = []
        for i in range(self.epoch):
            print('Epoch ', i+1)
            # 미니배치
            batch_indice = np.random.choice(len(X), batch_size)
            X_batch = X[batch_indice]
            Y_batch = Y[batch_indice]
            # 최적화
            for layer in self.layers:
                params = layer.get_params()
                # 경사하강법
                목표함수 = lambda params: self.compute_loss(
                    X_batch, y_batch)
                dW = numerical_gradient_batch(목표함수, params)
                params -= dW * self.학습률
            loss = self.compute_loss(X, y)
            loss_history.append(loss)
        return loss_history
```
최종본은 [이건](./neuralnet.py)데... 
근데 잘 작동안함 ㅋ 뭐하자는건지! 스스로 해결해보실?<p />
이용은 이렇게 (MNIST 데이터)
```python
model = FeedForwardNet(
    학습률=0.1, 학습횟수=10, 손실함수=cross_entropy)
model.add(Layer(784, 50, sigmoid))
model.add(Layer(50, 100, sigmoid))
model.add(Layer(100, 10, softmax))

Y_train = pd.get_dummies(y_train).values.astype('float32')
loss_history = model.fit(X_train, Y_train, batch_size=100)
```
## 4일차
### Tensorflow
케라스가 짱이니 케라스를 쓰겠습니다.
### Keras
설치는 Anaconda Prompt에서 conda install keras하면됨.
```python
import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()

model.add(Dense(50, input_shape=(784,), activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

from keras.optimizers import SGD

model.compile(
    loss='categorical_crossentropy', optimizer=SGD(lr=0.1),
    metrics=['accuracy']
)
model.summary()   # 모델 요약한거 보기

history = model.fit(
    X_train, Y_train, batch_size=128, epochs=200,
    validation_split=0.2
)                 # fit!
```
훈련 결과 평가 및 적용
```python
훈련결과 = pd.DataFrame(history.history)
훈련결과[['loss', 'val_loss']].plot()

score = model.evaluate(X_test, Y_test)
print('Loss: {0}, Acc.: {1}'.format(*score))  # 0.94정도 나옵니다

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)  # 젤 높은 값으로 추정숫자 선택
```
### 합성곱 신경망(CNN)
#### 구현 (MNIST 대상)
```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten

# 좀더 간결하게 코딩하는 꿀팁
model = Sequential([
  # 1층
  Conv2D(20, input_shape=(28, 28, 1), kernel_size=5, padding='same', activation='relu'),
  MaxPooling2D(pool_size=(2,2), strides=(2,2)),
  # 2층
  Conv2D(50, kernel_size=5, padding='same', activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  # 3층 (출력 준비층)
  Flatten(),
  Dense(500, activation='relu'),
  # 출력층
  Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X_train.reshape(-1, 28, 28, 1)    # CNN은 이미지같은 행렬형태의 데이터에 적합
X_test = X_test.reshape(-1, 28, 28, 1)

history = model.fit(X_train, Y_train, 
                    batch_size=128, epochs=20, 
                    validation_split=0.2)
```
#### 구현 (cifar10 대상)
우선 cifar10데이터 가져와서 전처리하는거
```python
from deepy.dataset import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load('data/keras/cifar-10-batches-py/')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = pd.get_dummies(y_train).values.astype('float32')
from keras.utils import np_utils
Y_test = np_utils.to_categorical(y_test)
```
다음은 모델(conv2d + conv2d + MaxPooling + dense + dense), Dropout도 들어감
```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout

model = Sequential([
  # 1층
  Conv2D(32, kernel_size=3, padding='same', input_shape=(32, 32, 3), activation='relu'),
  Conv2D(32, kernel_size=3, padding='same', activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),
  # 2층
  Conv2D(64, kernel_size=3, padding='same', activation='relu'),
  Conv2D(64, kernel_size=3, padding='same', activation='relu'),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),
  # 3층 (출력 준비층)
  Flatten(),
  Dense(512, activation='relu'),
  Dropout(0.5),
  # 출력층
  Dense(10, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train.shape, Y_train.shape    # 이미 잘 되어있죠?

history = model.fit(X_train, Y_train, 
                    batch_size=100, epochs=20, 
                    validation_split=0.2)
```
## 5일차
### Keras 고급 활용 (이미지 최신모델 활용)
VGG16같은거 keras자체에서 모델 제공됨. import해서 사용가능
#### VGG16
```python
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

vgg16 = VGG16(weights=None)
vgg16.summary()

vgg16.load_weights('data/keras/vgg16_weights_tf_dim_ordering_tf_kernels.h5') 
# 굳이파일로 안해도(파라미터전달안하면) 걍 웹에서 다운가능
```
다음은 모찌(개이름)로 실제 추론을 해볼건데요
```python
from keras.preprocessing import image

img = image.load_img('data/mozzi.jpg', target_size=(224, 224))
x = image.img_to_array(img)
X = np.array([x])
X = preprocess_input(X)

Y_pred = vgg16.predict(X)
y_pred = np.argmax(Y_pred, axis=1)
array([644], dtype=int64)   # 결과로 644가 나왔으니 확인해볼까요! 무려 품종까지 detect!
```
#### 논문에 소개된 모델 설명만 가지고 구현하기
여기보시면 cifar-10 accuracy 랭크가 있어요
(http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)
지금 2등하고있는 친구 논문을 보면 모델이 나와있죠 우린 C로 한번 만들어보죠
(https://arxiv.org/pdf/1412.6806.pdf)
```python
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.pooling import AveragePooling2D

model = Sequential()

model.add(Conv2D(96, kernel_size=(3,3), padding='same',                   
                 activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(96, kernel_size=(3,3), padding='same', 
                 activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(96, kernel_size=(3,3), padding='same',
                 strides=(2,2),
                 activation='relu'))

model.add(Conv2D(192, kernel_size=(3,3), padding='same',                   
                 activation='relu'))
model.add(Conv2D(192, kernel_size=(3,3), padding='same', 
                 activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(192, kernel_size=(3,3), padding='same',
                 strides=(2,2),
                 activation='relu'))

model.add(Conv2D(192, kernel_size=(3,3), padding='same',                   
                 activation='relu'))
model.add(Conv2D(192, kernel_size=(1,1), padding='same',                   
                 activation='relu'))
model.add(Conv2D(10, kernel_size=(1,1), padding='same', 
                 activation='relu'))

model.add(AveragePooling2D(pool_size=(6,6)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
history = model.fit(
    X_train, Y_train, batch_size=100, epochs=20, 
    validation_split=0.2)
```
### 재귀 신경망(RNN)
RNN은 1991년에 나왔고, 시계열 데이터나 음성처럼 시간의 흐름에 따른 예측에 좋아요!
### 구현 (airline 고객 데이터 대상)
```python
airline = pd.read_csv('data/international-airline-passengers.csv')  # 데이터 보시고
airline = airline.set_index('Month')  # 정리해줘야겠죠?
airline = airline.dropna()            # 없는 값 (NaN) 제거해야겠죠?

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(airline)  # 전처리(MinMaxScaling) 필요합니다잉

X = data[:-1]
y = data[1:]
len(X) == len(y)  # 우리는 예측을 할거라서요 이렇게 하나차이나게, 길이는 같게해줘야함!

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False)  # 맨날 하던 split인데 시계열 데이터니까 섞진말고!
X = X.reshape(-1, 1, 1)   # X의 형상은 (samples, time steps, features)
```
이제 대망의 모델 만들기
```python
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=1)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

from sklearn.metrics import r2_score
train_score = r2_score(y_train.flatten(), y_pred_train.flatten())
test_score = r2_score(y_test.flatten(), y_pred_test.flatten())
train_score, test_score   # 처참하네요? 그래도 눈으로 확인해보면 그정돈아님

xs_train = np.arange(len(X_train))
xs_test = np.arange(len(X_train), len(X))
plt.plot(scaler.inverse_transform(y))
plt.plot(xs_train, scaler.inverse_transform(y_pred_train))
plt.plot(xs_test, scaler.inverse_transform(y_pred_test))    # 그쵸?
```
나중에 심심하면 아래것들도 해보세요
```python
shampoo = pd.read_csv('data/shampoo.csv')
stocks = pd.read_csv('data/stock_px.csv')
pollution = pd.read_csv(
    'data/uci-ml/beijing-pm25.csv',
    index_col=0, 
    parse_dates=[['year', 'month', 'day', 'hour']]
)
```
끝.
