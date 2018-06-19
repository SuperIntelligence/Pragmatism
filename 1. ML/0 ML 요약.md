# 학습일정 및 필기 링크 
## [1일차](./Day%201/Day%201.md)
* 환경구성
* 파이썬 기본
* numpy, pandas
## [2일차](./Day%202/Day%202.md)
* MCP 뉴런
* 퍼셉트론
* scikit-learn
* kNN
## [3일차](./Day%203/Day%203.md)
* Linear Model
* Decision Tree
## [4일차](./Day%204/Day%204.md)
* Decision Tree (Ensemble)
* SVM
* 

결국 진도대로 나가지도 못했기 때문에 일단 한것만 써놓겠습니다.

# 교재
파이썬 라이브러리를 활용한 머신러닝
(http://www.yes24.com/24/goods/42806875?scode=032&OzSrank=3) <p />
교재 많이 따라갑니다. 일부 건너뛰고, 학습데이터를 바로 다운받는 대신 csv파일을 사용하는 정도.
(pandas library의 read_csv()함수 사용)

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
### scikit-learn
1일차의 X, y에 이어서. 학습데이터와 검증데이터를 3:1 비율로 나눠준다.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```
### kNN
#### Classifier
```python
from sklearn.neighbors import KNeighborsClassifier

이웃수 = 1 # 반복문으로 이웃수를 바꿔가며 score가 괜찮게 나오는 모델을 고르면 된다.
model = KNeighborsClassifier(n_neighbors=이웃수)

model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))
```
score()는 얼마나 맞췄는지를 1점만점으로 보여준다.
#### Regressor
```python
from sklearn.neighbor import KNeighborsRegressor

이웃수 = 1 # 반복문으로 이웃수를 바꿔가며 score가 괜찮게 나오는 모델을 고르면 된다.
model = KNeighborsRegressor(n_neighbors=이웃수)

model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))
```
## 3일차
### Linear Model
#### Classifier
#### Regressor

### Decision Tree
## 4일차
### Random Forest
앙상블(Ensemble) 기법은 복수 개의 모델을 수행해 학습 결과를 결합, 보다 좋은 성능을 내고자 하는 방법이다. <p />
Random Forest는 Decision Tree 여러개로 구성된 앙상블 모델이다.
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
```
하이퍼 파라미터는 n_estimators다.
### SVM
```python
from sklearn.svm import SVC

model = SVC().fit(X_train, y_train)
```
결과 잘 안나올거다. 왜냐? SVM류는 특별히 X를 **전처리**해주어야한다(특징값들의 범위에 민감하다나). <p />
보통 MinMaxScaler로 모든 데이터를 0~1사이의 값으로 변환해준다.
### 기타
#### preprocessing(전처리)
##### MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Xmm = scaler.fit_transform(X)
print(DataFrame(Xmm).head(5)) # 확인. 요렇게 바뀐다.

Xmm_train, Xmm_test, y_train, y_test = train_test_split(Xmm, y, stratify=y)

model = SVC().fit(Xmm_train, y_train) # 이렇게 변환된 X로 학습 및 검증하면 된다.
```
##### 차원축소
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

```
#### encoding(인코딩)
##### one-hot encoding
안녕
#### pickle 사용법
##### 짠지담그기
```python
import pickle

with open('trained_models.pkl', 'wb') as fp:
  pickle.dump({'model1': model1, 'model2': model2}, fp) # fit()된 친구들
```
걍 아무 오브젝트 묶어서 구조체 만들어서 파일로 저장하는거다. .pkl파일 옮기면된다.
##### 짠지꺼내먹기
```python
import pickle

with open('trained_models.pkl', 'rb') as fp:
  models = pickle.load(fp)

models['model1'].score(X_test, y_test) # 요렇게 갖다쓰면된다.
```
끝.
