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