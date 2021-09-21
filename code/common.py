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
               
    def fit(self, X: np.ndarray, t: np.ndarray, SVDfit=True) -> np.ndarray:
        if SVDfit:
            self.betas = SVDinv(X.T @ X) @ X.T @ t
        else:
            self.betas = np.linalg.pinv(X.T @ X) @ X.T @ t
        
        

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

    
def prepare_data(X: np.ndarray, t: np.ndarray, test_size=0.2, shuffle=True, scale_X= False, scale_t= False, random_state=None)-> np.ndarray:    
    """[summary]

    Args:
        X (np.ndarray): Design Matrix
        t (np.ndarray): Target vector
        test_size (float, optional): [description]. Defaults to 0.2.
        shuffle (bool, optional): [description]. Defaults to True.
        scale_X (bool, optional): [description]. Defaults to False.
        scale_t (bool, optional): [description]. Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    # split in training and test data
    if random_state is None:
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, shuffle=shuffle)
    else:
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, shuffle=shuffle, random_state=random_state)
        
    # Scale data        
    if(scale_X):
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
    if(scale_t):
        t_train = np.expand_dims(t_train,axis=1)
        t_test = np.expand_dims(t_test,axis=1)
        t_scaler = StandardScaler()
        t_scaler.fit(t_train)
        t_train = t_scaler.transform(t_train)
        t_test = t_scaler.transform(t_test)

    return X_train, X_test, t_train, t_test

"""
def FrankeFunction(x: float ,y: float) -> float:

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
"""


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


def bias_error_var(t_true, t_pred):
    error = np.mean( np.mean((t_true - t_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (t_true - np.mean(t_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(t_pred, axis=1, keepdims=True) )
    return error, bias, variance


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


# TODO: The methods below are temporary

def FrankeFunction(x, y):
    b1 = -((9*x - 2)**2)/4 - ((9*y - 2)**2)/4
    b2 = -((9*x + 1)**2)/49 - ((9*y + 1)**2)/10
    b3 = -((9*x - 7)**2)/4 - ((9*y - 3)**2)/4
    b4 = -((9*x - 4)**2) - ((9*y - 7)**2)
    
    a1 = 3/4 * np.exp(b1)
    a2 = 3/4 * np.exp(b2)
    a3 = 1/2 * np.exp(b3)
    a4 = -1/5 * np.exp(b4)
    return a1 + a2 + a3 + a4# + 0.25 * np.random.randn(len(x),)

def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


if __name__ == '__main__':
    print("Import this file as a package")
    
    