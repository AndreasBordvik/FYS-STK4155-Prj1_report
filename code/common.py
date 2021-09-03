import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as R2
from sklearn.preprocessing import StandardScaler


class Regression():
    def __init__(self):
        self.betas = None
        
    def fit(self, X, y):
        pass
        
    def get_all_betas(self):
        return self.betas
    
    def predict(self, X):        
        y_hat = X @ self.betas
        return y_hat    


class OLS(Regression):
    def __init__(self):
        super().__init__()
        #super().__init__(self)
        self.lambda_val = 1 # Lambda value
        
    def fit(self, X, y):
        X_T_X = (X.T @ X) 
        X_T_X += self.lambda_val * np.eye(X_T_X.shape[0]) # Preventing the singular matix issue
        self.betas = np.linalg.inv(X_T_X) @ X.T @ y
        

class LinearRegression(OLS):
    def __init__(self):
        super().__init__()
        

class RidgeRegression(Regression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y, lambda_val):
        pass 
        
         
class LassoRegression(Regression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y, lambda_val):
        pass 
        

def design_matrix(x, pol_degree):
        X = np.zeros((x.shape[0], pol_degree))
        X[:,0] = 1.0
        x = x.flatten()
        for i in range(1, X.shape[1]):
            X[:,i] = x ** i
            
        
        return X

    
def prepare_data(x, y, pol_degree, test_size=0.2, shuffle=True, scale= True):    
    X = design_matrix(x, pol_degree) 
    
    
    
    # split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=shuffle)
    
    # Scale data        
    if(scale):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    
    return X_train, X_test, y_train, y_test

    
def franke_function(x,y):
    pass




"""
def evaluate(targets, predictions):
    targets = targets.flatten()
    predictions = predictions.flatten()
    MSE_score = MSE(targets, predictions) # Can we use sklearn or de we have to write it from scratch?
    R2_score = R2(targets, predictions) # Can we use sklearn or de we have to write it from scratch?
    bias = None
    variance = None
    return MSE_score, R2_score, bias, variance
"""    


if __name__ == '__main__':
    print("Import this file as a package")
    
    