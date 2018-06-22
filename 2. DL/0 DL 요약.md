# 학습일정
## 1일차
* 환경설정
* Numpy, Matplotlib, Pandas
## 2일차
* MCP 뉴런
* 퍼셉트론
* 신경망
  - 다차원 배열 계산
  - 신경망 구현
  - 출력층
  - 신경망 적용
  - 손실 함수
  - 최적화
## 3일차
* 신경망 학습
  - 오차 역전파
  - 계층 구현
  - 활성화 함수 계층 구현
  - Affine/Softmax 계층 구현
  - 오차역전파 구현
  - 매개변수 상신
## 4일차
* Tensorflow
* Keras
* 합성곱 신경망(CNN)
  - 구현과 활용
## 5일차
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

안녕

## 4일차
### Tensorflow
### Keras
케라스가 짱이니 케라스를 쓰겠습니다.
### 합성곱 신경망(CNN)
  - 구현과 활용
## 5일차
### Keras 고급 활용
### 이미지 최신모델 활용
VGG16같은거 keras자체에서 모델 제공됨. import해서 사용가능
### 재귀 신경망(RNN)
  - 구현과 활용
