# 학습일정
## 1일차
* 환경설정
* Numpy, Matplotlib, Pandas
* MCP 뉴런
* 퍼셉트론
## 2일차
* 신경망
 * 다차원 배열 계산
 * 신경망 구현
 * 출력층
 * 신경망 적용
 * 손실 함수
 * 최적화
## 3일차
* 신경망 학습
 * 오차 역전파
 * 계층 구현
 * 활성화 함수 계층 구현
 * Affine/Softmax 계층 구현
 * 오차역전파 구현
 * 매개변수 상신
## 4일차
* Tensorflow
* Keras
* 합성곱 신경망(CNN)
 * 구현과 활용
## 5일차
* Keras 고급 활용
* 재귀 신경망(RNN)
 * 구현과 활용
 
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

def BOX(x1, x2, w1, w2, b):
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
    return BOX(x1, x2, w1, w2, b)
```
#### NAND 게이트
```python
def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    return BOX(x1, x2, w1, w2, b)
```
#### OR 게이트
```python
def OR(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.2
    return BOX(x1, x2, w1, w2, b)
```
### 
