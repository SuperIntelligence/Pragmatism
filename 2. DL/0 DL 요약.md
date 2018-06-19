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
 
 정리를 하면서 올려야 겠네요!
 
# 요약
## 1일차
### 환경구성
필요한건 python(3.x), 'numpy, matplotlib, pandas' 3종셋트, scikit-learn (머신러닝 라이브러리임) <p />
이정도인데, 결국 anaconda 설치하면 알아서 해결됨. (https://anaconda.org/anaconda)
### 파이썬 기본
linst, range, 기본적인 함수와 반복문 사용법 등
### numpy, matplotlib, pandas
#### numpy
np.array(): numpy에서 사용하는 ndarray형으로 변환 <p />
이후 numpy에서 제공하는 각종 메소드들, <p />
특별한 원소접근(팬시 색인, 불리언 색인) 등 사용가능
#### matplotlib
%matplotlib inline하면 plot만 해도 show 자동 <p />
matplotlib 간단한 사용법, 스타일 설정(색상, 선스타일)
#### pandas
pd.read_csv(): csv파일을 pandas에서 제공하는 dataframe형으로 변환 <p />
이후 pandas에서 제공하는 각종 메소드들 이용가능. <p />
(보통 그냥 .head(5)로 확인만 하고 y, X를 .values() 이용해서 ndarray로 삽입)
### 정리
고로 일반적으로 데이터는 다음과 같이 처리한다.
```python
import numpy as np
import pandas as pd

data = pd.read_csv('data/wdbc.data')
print(data.head(5), data.shape) # 한번 확인

y = data['class']
X = data.values[:,1:].astype('float32')
```

## 2일차
### MCP 뉴런
```python
import numpy as np

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
import numpy as np

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
import numpy as np
import pandas as pd

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
### 
