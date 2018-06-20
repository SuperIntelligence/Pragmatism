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

MCP 뉴런과 퍼셉트론은 DL에서 진도가 이어지기 때문에 [DL 요약에서 해당부분](./../2.%20DL/0%20DL%20요약.md) 참조하시면 됩니다.

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
#### Classification
```python
from sklearn.neighbors import KNeighborsClassifier

이웃수 = 1 # 반복문으로 이웃수를 바꿔가며 score가 괜찮게 나오는 모델을 고르면 된다.
model = KNeighborsClassifier(n_neighbors=이웃수)

model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))
```
score()는 얼마나 맞췄는지를 1점만점으로 보여준다.
#### Regression
```python
from sklearn.neighbor import KNeighborsRegressor

이웃수 = 1 # 반복문으로 이웃수를 바꿔가며 score가 괜찮게 나오는 모델을 고르면 된다.
model = KNeighborsRegressor(n_neighbors=이웃수)

model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))
```
## 3일차
### Linear Model
#### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
```
#### Ridge Regression
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0).fit(X_train, y_train)    # hyperparameter: alpha
```
하이퍼 파라미터를 바꿔가며 적절한 모델을 찾는 방법은 다음과 같다.
```python
scores = []
weights = []
alpha_range = [0.01, 0.1, 1., 10., 100., 1000.]
for alpha in alpha_range:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)    
    weights.append(ridge.coef_)
    train_score = ridge.score(X_train, y_train)
    test_score = ridge.score(X_test, y_test)
    scores.append((train_score, test_score))
    
훈련평가 = pd.DataFrame(
    scores, 
    index=alpha_range, 
    columns=['train', 'test'])
훈련평가.plot(
    logx=True, ylim=(0.3, 1.0),
    style=['go--', 'ro--'])
```
과대적합과 과소적합이 발생하지 않는 적절한 값을 고르면 된다. 이 방식은 다른 모든 기계학습모델에 적용할 수 있다.
#### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1).fit(X_train, y_train)  # hyperparameter: C
```
C 값은 보통 [0.01, 0.1, 1., 10., 100., 1000.]
### Decision Tree
#### Classification
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)  # hyperparameter: max_depth
```
max_depth는 보통 list(range(1, 11))
#### Regression
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor().fit(X_train, y_train)
```
## 4일차
### Ensemble
앙상블(Ensemble) 기법은 복수 개의 모델을 수행해 학습 결과를 결합, 보다 좋은 성능을 내고자 하는 방법이다. <p />
Random Forest와 Gradient Boosting은 Decision Tree 여러개로 구성된 앙상블 알고리즘이다.
#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)  # hyperparameter : n_estimators
```
#### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, max_depth=1).fit(X_train, y_train)
```
### SVM
```python
from sklearn.svm import SVC

model = SVC(C=1000).fit(X_train, y_train)   # hyperparameter: C
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

model = SVC(C=1000).fit(Xmm_train, y_train) # 이렇게 변환된 X로 학습 및 검증하면 된다.
```
대체로 선형모델, 그를 기반으로 한 딥러닝 등이 Scale에 민감하므로 꼭 Scaling을 해주어야 한다.
##### 차원축소, 파이프라인
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('model', LogisticRegression())
]).fit(X_train, y_train)    # X_train을 Scaling 후, 2차원으로 축소하고, fit.
```
##### 차원축소(PCA)를 활용한 특성 중요도 평가
```python
pca = PCA(n_components=None).fit(Xstd_train)  # Standard Scaling한 데이터 넣음
  
특성변량기여도 = pca.explained_variance_ratio_
특성변량기여도 = Series(
    특성변량기여도, index=cancer.columns[1:])
특성변량기여도.sort_values().plot(kind='barh')
```
#### encoding(인코딩)
##### one-hot encoding
```python
import pandas as pd

pd.get_dummies(y)
```
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
