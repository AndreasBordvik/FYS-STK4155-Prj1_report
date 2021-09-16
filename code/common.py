import numpy as np
from numpy.core.defchararray import index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

class Regression():
    def __init__(self):
        self.betas = None
                
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        pass
        
    def get_all_betas(self):
        return self.betas
    
    def predict(self, X:np.ndarray) -> np.ndarray:        
        prediction = X @ self.betas
        return prediction


class OLS(Regression):
    def __init__(self):
        super().__init__()
               
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.betas = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        

class LinearRegression(OLS):
    def __init__(self):
        super().__init__()
        

class RidgeRegression(Regression):
    def __init__(self, lambda_val:float):
        super().__init__()
        self.lam = lambda_val
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: 
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]
            lambda_val (float): [description]

        Returns:
            np.ndarray: [description]
        """
        X_T_X = X.T @ X 
        X_T_X += self.lam * np.eye(X_T_X.shape[0]) # beta punishing and preventing the singular matix
        self.betas = np.linalg.inv(X_T_X) @ X.T @ y 
        
         
class LassoRegression(Regression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X: np.ndarray, y: np.ndarray, lambda_val:float) -> np.ndarray: 
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]
            lambda_val (float): [description]

        Returns:
            np.ndarray: [description]
        """
        X_T_X = X.T @ X 
        X_T_X += lambda_val * np.eye(X_T_X.shape[0]) # beta punishing and preventing the singular matix
        self.betas = np.linalg.inv(X_T_X) @ X.T @ y 
        

def design_matrix(x: np.ndarray, features:int)-> np.ndarray:
    """design_matrix

    Args:
        x (np.ndarray): [description]
        pol_degree (int): [description]

    Returns:
        np.ndarray: [description]
    """
    X = np.zeros((x.shape[0], features))
    x = x.flatten()
    for i in range(1,X.shape[1]+1):
        X[:,i-1] = x ** i
    return X

    
def prepare_data(x: np.ndarray, y: np.ndarray, features:int, test_size=0.2, shuffle=True, scale_x= True, scale_y= False, intercept=False, X = None)-> np.ndarray:    
    """[summary]

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        pol_degree (int): [description]
        test_size (float, optional): [description]. Defaults to 0.2.
        shuffle (bool, optional): [description]. Defaults to True.
        scale (bool, optional): [description]. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if (X is None):
        X = design_matrix(x, features)

    # split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=shuffle)
        
    # Scale data        
    if(scale_x):
        x_scaler = StandardScaler()
        x_scaler.fit(X_train)
        X_train = x_scaler.transform(X_train)
        X_test = x_scaler.transform(X_test)
    
    if(scale_y):
        y_scaler = StandardScaler()
        y_scaler.fit(y_train)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test)

    if intercept: # Adding intercept to the data. Default false
        X_train = np.expand_dims(X_train, axis=0)
        X_train[:,0] = 1
        X_test = np.expand_dims(X_test, axis=0)
        X_test[:,0] = 1
    
    return X_train, X_test, y_train, y_test

    

def FrankeFunction(x: float ,y: float) -> float:
    """[summary]

    Args:
        x (float): [description]
        y (float): [description]

    Returns:
        float: [description]
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x:np.ndarray, y:np.ndarray, n:int )->np.ndarray:
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def FrankeFunctionMeshgrid() -> np.ndarray:
    # Making meshgrid of datapoints and compute Franke's function
    n = 5
    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    z = FrankeFunction(x, y)
    X = create_X(x, y, n=n)
    return X  


def plot_franke_function():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

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
    
    