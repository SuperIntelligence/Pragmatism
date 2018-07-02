# %load neuralnet.py
import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    y = exp_a / np.sum(exp_a)
    return y

def cross_entropy(y_pred, y):
    "batch_size: mini batch 크기"
    batch_size = len(y)
    return -np.sum(y * np.log(y_pred)) / batch_size

def numerical_gradient_batch(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    
    grad = np.zeros_like(X)
    # 각 샘플에 대해 기울기 산출
    for idx, x in enumerate(X):
        grad[idx] = numerical_gradient(f, X)
    return grad    

class Layer:
    def __init__(self, 입력수, 출력수, 활성화_함수):
        self.W = np.random.randn(입력수, 출력수)
        self.b = np.random.randn(출력수)
        self.activation = 활성화_함수
        
    def output(self, X):
        z = np.dot(X, self.W) + self.b
        y = self.activation(z)
        return y
    
    def get_params(self):
        return np.vstack([self.W, self.b])

class FeedForwardNet:
    def __init__(self, 학습률, 학습횟수, 손실함수):
        self.layers = []
        self.learning_rate = 학습률
        self.epoch = 학습횟수
        self.loss_func = 손실함수
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        """순전파"""
        layer_input = X
        for layer in self.layers:
            layer_input = layer.output(layer_input)
        
        y = layer_input
        return y
    
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
                목표함수 = lambda params: self.compute_loss(X, y)
                dW = numerical_gradient_batch(목표함수, params)
                params -= dW * self.학습률
            loss = self.compute_loss(X, y)
            loss_history.append(loss)
        return loss_history