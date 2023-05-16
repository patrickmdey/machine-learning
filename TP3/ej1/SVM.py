import numpy as np

class SVM:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
        self.b = 0.0

    def train(self, X, y, C, learning_rate, epochs=10000):
        learning_rate = learning_rate * np.exp(-0.0001 * epochs)
        for _ in range(epochs):
            for _ in range(len(X)):
                i = np.random.randint(0, len(X))
                t = np.dot(y[i], np.dot(X[i], self.weights) + self.b) #TODO: append 1 for b in X[i] ?

                if t < 1:
                    self.weights = self.weights - learning_rate * (self.weights - C * np.dot(y[i], X[i]))
                    self.b = self.b + learning_rate *  (C * y[i])
                else:
                    self.weights = self.weights - learning_rate * self.weights
            
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.b 
        return np.where(z >= 0, 1, -1)
    
    def calculate_margin(self):
        return 1/ np.linalg.norm(self.weights)
        # distances = np.abs(np.dot(X, self.weights) + self.bias) / np.linalg.norm(self.weights)
        # margin = np.min(distances)
        # return margin