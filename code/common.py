import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from tqdm import tqdm
from numpy.core.defchararray import index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.linear_model as lm
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/" # Path for data ment for the report
REPORT_FIGURES = "../figures/" # Path for figures ment for the report
SEED_VALUE = 4155
EX1 = "EX1_"; EX2 = "EX2_"; EX3 = "EX3_"; EX4 = "EX4_"; EX5 = "EX5_"
EX6 = "EX6_"; EX6_1 = f"{EX6}{EX1}"; EX6_2 = f"{EX6}{EX2}"; 
EX6_3 = f"{EX6}{EX3}"; EX6_4 = f"{EX6}{EX4}"; EX6_5 = f"{EX6}{EX5}"

class Regression():
    def __init__(self):
        self.betas = None
        self.X_train = None
        self.t_train = None
        self.t_hat_train = None
        self.param = None
        self.param_name = None
        self.SVDfit = None
        self.SE_betas = None
        self.keep_intercept = None
                
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @property      
    def get_all_betas(self):
        return self.betas
    
    def predict(self, X:np.ndarray) -> np.ndarray:        
        #print("from predict: self.keep_intercept:",self.keep_intercept)
        if(self.keep_intercept == False):
            #print(f"Predict: removing intercept")
            X = X[:, 1:]
        #print("betas.shape in predict:",self.betas.shape)
        #print("X.shape in predict:",X.shape)
        prediction = X @ self.betas
        return prediction

    @property
    def SE(self):
        var_hat = (1./self.X_train.shape[0]) * np.sum((self.t_train - self.t_hat_train)**2)

        if self.SVDfit:
            invXTX_diag = np.diag(SVDinv(self.X_train.T @ self.X_train))
        else:
            invXTX_diag = np.diag(np.linalg.pinv(self.X_train.T @ self.X_train))
        return np.sqrt(var_hat * invXTX_diag) 

    def summary(self):
        # Estimated standard error for the beta coefficients
        N, P = self.X_train.shape
        SE_betas = self.SE
        
        # Calculating 95% confidence intervall
        CI_lower_all_betas = self.betas - (1.96 * SE_betas)
        CI_upper_all_betas = self.betas + (1.96 * SE_betas)
        
        # Summary dataframe
        params = np.zeros(self.betas.shape[0]); params.fill(self.param)
        coeffs_df = pd.DataFrame.from_dict({f"{self.param_name}" :params,
                                    "coeff_name": [f"b_{i}" for i in range(0,self.betas.shape[0])],
                                    "coeff_value": np.round(self.betas, decimals=4),
                                    "std_error": np.round(SE_betas, decimals=4),
                                    "CI_lower":np.round(CI_lower_all_betas, decimals=4), 
                                    "CI_upper":np.round(CI_upper_all_betas, decimals=4)},
                                    orient='index').T
        return coeffs_df

class OLS(Regression):
    def __init__(self, degree = 1, param_name="degree"):
        super().__init__()
        self.param = degree
        self.param_name = param_name
               
    def fit(self, X: np.ndarray, t: np.ndarray, SVDfit=True, keep_intercept=True) -> np.ndarray:
        self.SVDfit = SVDfit
        self.keep_intercept = keep_intercept
        if keep_intercept == False:
            X = X[:, 1:]

        self.X_train = X
        self.t_train = t
        
        if SVDfit:
            self.betas = SVDinv(X.T @ X) @ X.T @ t
        else:
            self.betas = np.linalg.pinv(X.T @ X) @ X.T @ t  
        self.t_hat_train = X @ self.betas
        #print("betas.shape in train before squeeze:",self.betas.shape)
        self.betas = np.squeeze(self.betas)
        #print("betas.shape in train after squeeze:",self.betas.shape)
        return self.t_hat_train
    
class LinearRegression(OLS):
    def __init__(self):
        super().__init__()
        
class RidgeRegression(Regression):
    def __init__(self, lambda_val = 1, param_name="lambda"):
        super().__init__()
        self.param = self.lam = lambda_val
        self.param_name = param_name
                
    def fit(self, X: np.ndarray, t: np.ndarray, SVDfit=True, keep_intercept=True) -> np.ndarray: 
        self.SVDfit = SVDfit
        self.keep_intercept = keep_intercept
        if keep_intercept == False:
            X = X[:, 1:]
        self.X_train = X
        self.t_train = t
        Hessian = X.T @ X
        Hessian += self.lam * np.eye(Hessian.shape[0]) # beta punishing and preventing the singular matix
                
        if SVDfit:
            self.betas = SVDinv(Hessian) @ X.T @ t
        else:
            self.betas = np.linalg.pinv(Hessian) @ X.T @ t
        self.t_hat_train = X @ self.betas
        #print(f"Betas.shape in Ridge before:{self.betas.shape}")
        self.betas = np.squeeze(self.betas)
        #print(f"Betas.shape in Ridge after:{self.betas.shape}")
        return self.t_hat_train 
        
class LassoRegression(Regression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X: np.ndarray, y: np.ndarray, lambda_val:float) -> np.ndarray: 
        X_T_X = X.T @ X 
        X_T_X += lambda_val * np.eye(X_T_X.shape[0]) # beta punishing and preventing the singular matix
        self.betas = np.linalg.inv(X_T_X) @ X.T @ y 
        

def design_matrix(x: np.ndarray, features:int)-> np.ndarray:
    X = np.zeros((x.shape[0], features))
    x = x.flatten()
    for i in range(1,X.shape[1]+1):
        X[:,i-1] = x ** i
    return X

    
def prepare_data(X: np.ndarray, t: np.ndarray, test_size=0.2, shuffle=True, scale_X= False, scale_t= False, zero_center = False, random_state=SEED_VALUE)-> np.ndarray:    
    # split in training and test data
    if random_state is None:
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, shuffle=shuffle)
    else:
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, shuffle=shuffle, random_state=random_state)
        
    # Scale data        
    if(scale_X):
        if zero_center:
            X_train = manual_scaling(X_train)
            X_test = manual_scaling(X_test)
        else:
            X_train, _ = standard_scaling(X_train)    
            X_test, _ = standard_scaling(X_test)    

    if(scale_t):
        if zero_center:
            t_train = manual_scaling(t_train)
            t_test = manual_scaling(t_test)
        else:
            t_train, _ = standard_scaling(t_train)
            t_test, _ = standard_scaling(t_test)
            
    return X_train, X_test, t_train, t_test

def manual_scaling(data):
    """
    Avoids the use of sklearn StandardScaler(), which also 
    divides the scaled value by the standard deviation.
    This scaling is essentially just a zero centering
    """
    return data - np.mean(data,axis=0)

def standard_scaling(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def min_max_scaling(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def FrankeFunction(x: float ,y: float) -> float:
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x:np.ndarray, y:np.ndarray, n:int)->np.ndarray:
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
"""
def create_X(x:np.ndarray, y:np.ndarray, n:int)->np.ndarray:
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)  # Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
            
    return X
"""

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

def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U),len(VT)))
    D = np.diag(s)
    #print(D)
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return V@(invD@UT)

def bootstrap(x, y, t, maxdegree, n_bootstraps, model='Linear', lmb=None):
    for degree in tqdm(range(maxdegree), desc = f"Looping trhough polynomials up to {maxdegree} with {n_bootstraps}: "):
        MSE_test = np.zeros(maxdegree)
        MSE_train = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)

        X = create_X(x,y, n=degree)
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=SEED_VALUE)
        t_test_ = np.reshape(t_test, newshape=(t_test.shape[0],1))
        t_train_ = np.reshape(t_train, newshape=(t_train.shape[0],1))
        X_train = standard_scaling(X_train)
        X_test = standard_scaling(X_test)

        """
        if model == 'Linear':
            model = lm.LinearRegression()
        elif model == 'Ridge':
            model = RidgeRegression(lmb)
        elif model == 'Lasso':
            model = Lasso(lmb)
        else:
            print(f"No valid model was chose, {model} is not a valid model")
        """
        #model = lm.LinearRegression() 
        model = OLS(degree=degree)
        t_hat_train, t_hat_test = bootstrapping(X_train, t_train, X_test, t_test, n_bootstraps, model)

        MSE_test[degree] = np.mean( np.mean((t_test_ - t_hat_test)**2, axis=1, keepdims=True))
        MSE_train[degree] = np.mean( np.mean((t_train_ - t_hat_train)**2, axis=1, keepdims=True))
        bias[degree] = np.mean((t_test - np.mean(t_hat_test, axis=1, keepdims=True))**2)
        variance[degree] = np.mean(np.var(t_hat_test, axis=1, keepdims=True))    
    return MSE_test, MSE_train, bias, variance

def bootstrapping(X_train, t_train, X_test, t_test, n_bootstraps, model, keep_intercept=False):
    t_hat_trains = np.empty((t_train.shape[0], n_bootstraps))
    t_hat_tests = np.empty((t_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X, t = resample(X_train, t_train)
        t_hat_train = model.fit(X, t, SVDfit=False, keep_intercept=keep_intercept)
        t_hat_test = model.predict(X_test)
        # Storing predictions
        t_hat_trains[:,i] = t_hat_train.ravel()
        t_hat_tests[:,i] = t_hat_test.ravel()
    return t_hat_trains, t_hat_tests


def plot_beta_errors_for_lambdas(summaries_df : pd.DataFrame(), degree):
    grp_by_coeff_df = summaries_df.groupby(["coeff_name"])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    i=0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        #display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff_value"].to_numpy().astype(np.float64)
        beta_SE = df["std_error"].to_numpy().astype(np.float64)
        
        # plot beta values
        #plt.plot(lambdas, beta_values, label=f"b{i}")
        plt.plot(lambdas, beta_values, label=fr"$\beta_{i}$$\pm SE$")
        #plt.plot(lambdas, beta_values)
        
        # plot std error
        plt.fill_between(lambdas, beta_values-beta_SE, beta_values+beta_SE, alpha = 0.2)
        
        # 95% CI
        #plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i+=1

    plt.title(f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ $\pm$ SE")
    plt.xscale("log")
    plt.legend(bbox_to_anchor = (1.05, 1.1))
    #plt.show()
    #plt.tight_layout()
    return fig


def plot_beta_CI_for_lambdas(summaries_df : pd.DataFrame(), degree):
    grp_by_coeff_df = summaries_df.groupby(["coeff_name"])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    i=0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        #display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff_value"].to_numpy().astype(np.float64)
        CI_lower = df["CI_lower"].to_numpy().astype(np.float64)
        CI_upper = df["CI_upper"].to_numpy().astype(np.float64)
        
        # plot beta values
        #plt.plot(lambdas, beta_values, label=f"b{i}")
        plt.plot(lambdas, beta_values, label=fr"$\beta_{i}$$\pm SE$")
        #plt.plot(lambdas, beta_values)
        
        # plot std error
        plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        
        # 95% CI
        #plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i+=1

    plt.title(f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ CI")
    plt.xscale("log")
    plt.legend(bbox_to_anchor = (1.05, 1.1))
    #plt.tight_layout()
    #plt.show()
    return fig
    

def plot_beta_errors(summaary_df : pd.DataFrame(), degree):
    betas = summaary_df["coeff_value"].to_numpy().astype(np.float64)
    SE = summaary_df["std_error"].to_numpy().astype(np.float64)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(f"Beta error OLS - degree{degree}")
    plt.xlabel(r"$\beta_i$")
    plt.ylabel("Beta values with Std error")
    plt.xticks(np.arange(summaary_df.shape[0]))
    plt.errorbar(np.arange(summaary_df.shape[0]), betas , yerr = SE, fmt = 'o', ms=4)
    #plt.tight_layout()
    #plt.show()
    return fig
    


if __name__ == '__main__':
    print("Import this file as a package")
    
    