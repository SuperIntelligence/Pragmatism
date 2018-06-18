# 학습일정 및 필기 링크 
## [1일차](./Day%201/Day%201.md)
* 환경구성
* 파이썬 기본
* numpy, pandas
## [2일차]("./Day%202/Day%202.md")
* MCP 뉴런
* 퍼셉트론
* scikit-learn
* k-NN
* linear model (regressor)
* linear model (classifier)
## [3일차](./Day_3/Day_3.md)
* decision tree
* decision tree (앙상블)
* SVM
* neural network
## [4일차](Day 4/Day 4.md)

결국 진도대로 나가지도 못했기 때문에 일단 한것만 써놓겠습니다.

# 교재
파이썬 라이브러리를 활용한 머신러닝
(http://www.yes24.com/24/goods/42806875?scode=032&OzSrank=3)

교재 많이 따라갑니다. 일부 건너뛰고, 학습데이터를 바로 다운받는 대신 csv파일을 사용하는 정도.
(pandas library의 read_csv()함수 사용)

# 요약
## 1일차
### 환경구성
필요한건 python(3.x), 'numpy, matplotlib, pandas' 3종셋트, scikit-learn (머신러닝 라이브러리임)
이정도인데, 결국 anaconda 설치하면 알아서 해결됨. (https://anaconda.org/anaconda)
### 파이썬 기본
linst, range, 기본적인 함수와 반복문 사용법 등
### numpy, matplotlib, pandas
#### numpy
np.array(): numpy에서 사용하는 ndarray형으로 변환
이후 numpy에서 제공하는 각종 메소드들,
특별한 원소접근(팬시 색인, 불리언 색인) 등 사용가능
#### matplotlib
%matplotlib inline하면 plot만 해도 show 자동
matplotlib 간단한 사용법, 스타일 설정(색상, 선스타일)
#### pandas
pd.read_csv(): csv파일을 pandas에서 제공하는 dataframe형으로 변환
이후 pandas에서 제공하는 각종 메소드들 이용가능.
(보통 그냥 .head(5)로 확인만 하고 y, X를 .values() 이용해서 ndarray로 삽입)

## 2일차
다음기회에^^ 
