import time
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
from sklearn.model_selection import KFold
from sklearn import linear_model


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
SEED_VALUE = 4155
EX1 = "EX1_"
EX2 = "EX2_"
EX3 = "EX3_"
EX4 = "EX4_"
EX5 = "EX5_"
EX6 = "EX6_"
EX6_1 = f"{EX6}{EX1}"
EX6_2 = f"{EX6}{EX2}"
EX6_3 = f"{EX6}{EX3}"
EX6_4 = f"{EX6}{EX4}"
EX6_5 = f"{EX6}{EX5}"


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

    def predict(self, X: np.ndarray) -> np.ndarray:
        # print("from predict: self.keep_intercept:",self.keep_intercept)
        if(self.keep_intercept == False):
            # print(f"Predict: removing intercept")
            X = X[:, 1:]
        # print("betas.shape in predict:",self.betas.shape)
        # print("X.shape in predict:",X.shape)
        prediction = X @ self.betas
        return prediction

    @property
    def SE(self):
        var_hat = (1./self.X_train.shape[0]) * \
            np.sum((self.t_train - self.t_hat_train)**2)

        if self.SVDfit:
            invXTX_diag = np.diag(SVDinv(self.X_train.T @ self.X_train))
        else:
            invXTX_diag = np.diag(np.linalg.pinv(
                self.X_train.T @ self.X_train))
        return np.sqrt(var_hat * invXTX_diag)

    def summary(self):
        # Estimated standard error for the beta coefficients
        N, P = self.X_train.shape
        SE_betas = self.SE

        # Calculating 95% confidence intervall
        CI_lower_all_betas = self.betas - (1.96 * SE_betas)
        CI_upper_all_betas = self.betas + (1.96 * SE_betas)

        # Summary dataframe
        params = np.zeros(self.betas.shape[0])
        params.fill(self.param)
        coeffs_df = pd.DataFrame.from_dict({f"{self.param_name}": params,
                                            "coeff_name": [f"b_{i}" for i in range(0, self.betas.shape[0])],
                                            "coeff_value": np.round(self.betas, decimals=4),
                                            "std_error": np.round(SE_betas, decimals=4),
                                            "CI_lower": np.round(CI_lower_all_betas, decimals=4),
                                            "CI_upper": np.round(CI_upper_all_betas, decimals=4)},
                                           orient='index').T
        return coeffs_df


class OLS(Regression):
    def __init__(self, degree=1, param_name="degree"):
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
        # print("betas.shape in train before squeeze:",self.betas.shape)
        self.betas = np.squeeze(self.betas)
        # print("betas.shape in train after squeeze:",self.betas.shape)
        return self.t_hat_train


class LinearRegression(OLS):
    def __init__(self):
        super().__init__()


class RidgeRegression(Regression):
    def __init__(self, lambda_val=1, param_name="lambda"):
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
        # beta punishing and preventing the singular matix
        Hessian += self.lam * np.eye(Hessian.shape[0])

        if SVDfit:
            self.betas = SVDinv(Hessian) @ X.T @ t
        else:
            self.betas = np.linalg.pinv(Hessian) @ X.T @ t
        self.t_hat_train = X @ self.betas
        # print(f"Betas.shape in Ridge before:{self.betas.shape}")
        self.betas = np.squeeze(self.betas)
        # print(f"Betas.shape in Ridge after:{self.betas.shape}")
        return self.t_hat_train


class LassoRegression(Regression):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
        X_T_X = X.T @ X
        # beta punishing and preventing the singular matix
        X_T_X += lambda_val * np.eye(X_T_X.shape[0])
        self.betas = np.linalg.inv(X_T_X) @ X.T @ y


def design_matrix(x: np.ndarray, features: int) -> np.ndarray:
    X = np.zeros((x.shape[0], features))
    x = x.flatten()
    for i in range(1, X.shape[1]+1):
        X[:, i-1] = x ** i
    return X


def prepare_data(X: np.ndarray, t: np.ndarray, test_size=0.2, shuffle=True, scale_X=False, scale_t=False, zero_center=False, random_state=SEED_VALUE) -> np.ndarray:
    # split in training and test data
    if random_state is None:
        X_train, X_test, t_train, t_test = train_test_split(
            X, t, test_size=test_size, shuffle=shuffle)
    else:
        X_train, X_test, t_train, t_test = train_test_split(
            X, t, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # Scale data
    if(scale_X):
        if zero_center:  # This should NEVER happen
            X_train = manual_scaling(X_train)
            X_test = manual_scaling(X_test)
        else:
            X_train, X_test = standard_scaling(X_train, X_test)

    if(scale_t):
        if zero_center:  # This should NEVER happen
            t_train = manual_scaling(t_train)
            t_test = manual_scaling(t_test)
        else:
            t_train, t_test = standard_scaling(t_train, t_test)

    return X_train, X_test, t_train, t_test


def manual_scaling(data):
    """
    Avoids the use of sklearn StandardScaler(), which also
    divides the scaled value by the standard deviation.
    This scaling is essentially just a zero centering
    """
    return data - np.mean(data, axis=0)


def standard_scaling(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled

def standard_scaling_single(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def min_max_scaling(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def FrankeFunction(x: float, y: float) -> float:
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def timer(func) -> float:
    """
    Simple timer that can be used as a decorator to time functions
    """
    def timer_inner(*args, **kwargs):
        t0: float = time.time()
        result = func(*args, **kwargs)
        t1: float = time.time()
        print(
            f"Elapsed time {1000*(t1 - t0):6.4f}ms in function {func.__name__}"
        )
        return result
    return timer_inner


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
    error = np.mean(np.mean((t_true - t_pred)**2, axis=1, keepdims=True))
    bias = np.mean((t_true - np.mean(t_pred, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(t_pred, axis=1, keepdims=True))
    return error, bias, variance


def plot_franke_function():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
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
        # Can we use sklearn or de we have to write it from scratch?
        MSE_score = MSE(targets, predictions)
        # Can we use sklearn or de we have to write it from scratch?
        R2_score = R2(targets, predictions)
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

    D = np.zeros((len(U), len(VT)))
    D = np.diag(s)
    # print(D)
    UT = np.transpose(U)
    V = np.transpose(VT)
    invD = np.linalg.inv(D)
    return V@(invD@UT)


@timer
def bootstrap(x, y, t, maxdegree, n_bootstraps, model, scale_X=False, scale_t=False):

    MSE_test = np.zeros(maxdegree)
    MSE_train = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    t_flat = t.ravel().reshape(-1, 1)

    for degree in tqdm(range(1, maxdegree+1), desc=f"Looping trhough polynomials up to {maxdegree} with {n_bootstraps}: "):
        X = create_X(x, y, n=degree)
        X_train, X_test, t_train, t_test = prepare_data(
            X, t_flat, test_size=0.2, shuffle=True, scale_X=scale_X, scale_t=scale_t, random_state=SEED_VALUE)

        t_hat_train, t_hat_test = bootstrapping(
            X_train, t_train, X_test, t_test, n_bootstraps, model, keep_intercept=True)

        MSE_test[degree-1] = np.mean(
            np.mean((t_test - t_hat_test)**2, axis=1, keepdims=True))
        MSE_train[degree-1] = np.mean(
            np.mean((t_train - t_hat_train)**2, axis=1, keepdims=True))
        bias[degree-1] = np.mean(
            (t_test - np.mean(t_hat_test, axis=1, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(t_hat_test, axis=1, keepdims=True))
    return MSE_test, MSE_train, bias, variance


def bootstrapping(X_train, t_train, X_test, t_test, n_bootstraps, model, keep_intercept=False):
    t_hat_trains = np.empty((t_train.shape[0], n_bootstraps))
    t_hat_tests = np.empty((t_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X, t = resample(X_train, t_train)
        t_hat_train = model.fit(
            X, t, SVDfit=False, keep_intercept=keep_intercept)
        t_hat_test = model.predict(X_test)
        # Storing predictions
        t_hat_trains[:, i] = t_hat_train.ravel()
        t_hat_tests[:, i] = t_hat_test.ravel()
    return t_hat_trains, t_hat_tests


def plot_beta_errors_for_lambdas(summaries_df: pd.DataFrame(), degree):
    grp_by_coeff_df = summaries_df.groupby(["coeff_name"])

    # plt.rcParams["figure.figsize"] = [7.00, 3.50]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i = 0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        # display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff_value"].to_numpy().astype(np.float64)
        beta_SE = df["std_error"].to_numpy().astype(np.float64)

        # plot beta values
        # plt.plot(lambdas, beta_values, label=f"b{i}")
        plt.plot(lambdas, beta_values, label=fr"$\beta_{i}$$\pm SE$")
        # plt.plot(lambdas, beta_values)

        # plot std error
        plt.fill_between(lambdas, beta_values-beta_SE,
                         beta_values+beta_SE, alpha=0.2)

        # 95% CI
        # plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i += 1

    plt.title(
        f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ $\pm$ SE")
    plt.xscale("log")
    if degree < 5:
        plt.rcParams["figure.autolayout"] = True
        plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.show()
    # plt.tight_layout()
    return fig


def plot_beta_CI_for_lambdas(summaries_df: pd.DataFrame(), degree):
    grp_by_coeff_df = summaries_df.groupby(["coeff_name"])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i = 0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        # display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff_value"].to_numpy().astype(np.float64)
        CI_lower = df["CI_lower"].to_numpy().astype(np.float64)
        CI_upper = df["CI_upper"].to_numpy().astype(np.float64)

        # plot beta values
        # plt.plot(lambdas, beta_values, label=f"b{i}")
        plt.plot(lambdas, beta_values, label=fr"$\beta_{i}$ with $CI$")
        # plt.plot(lambdas, beta_values)

        # plot std error
        plt.fill_between(lambdas, CI_lower, CI_upper, alpha=0.2)

        # 95% CI
        # plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i += 1

    plt.title(
        f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ with $CI_{95}$")
    plt.xscale("log")
    if degree < 5:
        plt.rcParams["figure.autolayout"] = True
        plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.tight_layout()
    # plt.show()
    return fig


def plot_beta_errors(summaary_df: pd.DataFrame(), degree):
    betas = summaary_df["coeff_value"].to_numpy().astype(np.float64)
    SE = summaary_df["std_error"].to_numpy().astype(np.float64)

    # Computing x-ticks
    x_ticks = ["1"]
    for i in range(1, degree+1):
        for k in range(i+1):
            x_ticks.append(f"({i-k})({k})")

    fig = plt.figure()
    ax = plt.axes()
    plt.title(f"Beta error OLS - degree{degree}")
    plt.xlabel(r"$\beta_i$ as power of x and y")
    plt.ylabel("Beta values with std error")
    ax.set_xticks(np.arange(summaary_df.shape[0]))
    ax.set_xticklabels(x_ticks)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize/plt.gcf().dpi*summaary_df.shape[0]+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.errorbar(
        np.arange(summaary_df.shape[0]), betas, yerr=SE, fmt='o', ms=4)
    # plt.tight_layout()
    # plt.show()
    return fig


def cross_val(k: int, model: str, X: np.ndarray, z: np.ndarray, lmb=None, shuffle=False) -> np.ndarray:
    """Function for cross validating on k folds. Scales data after split(standarscaler).

    Args:
        k (int): Number of folds
        model (str): Linear regression model
        X (np.ndarray): Design matrix
        z (np.ndarray): target values
        lmb (Optional): lambda value
        shuffle (boolean): deafault False. 

    Returns:
        np.ndarray: Scores of MSE on all k folds
    """
    if model == "Ridge":
        model = RidgeRegression(lambda_val=lmb)
    elif model == "Lasso":
        model = lm.Lasso(alpha=lmb)
    elif model == "OLS":
        model = OLS()

    else:
        "Provide a valid model as a string(Ridge/Lasso/OLS) "



    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=SEED_VALUE)
    scores_KFold = np.zeros(k)
    z = z.ravel()
    # scores_KFold idx counter
    j = 0
    for train_inds, test_inds in kfold.split(X, z):

        # get all cols and selected train_inds rows/elements:
        xtrain = X[train_inds, :]
        ytrain = z[train_inds]
        # get all cols and selected test_inds rows/elements:
        xtest = X[test_inds, :]
        ytest = z[test_inds]

        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain_scaled = scaler.transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        xtrain_scaled[:, 0] = 1
        xtest_scaled[:, 0] = 1

        if model == "Ridge":
            model.fit(xtrain_scaled, ytrain, keep_intercept=False)
        else:
            model.fit(xtrain_scaled, ytrain)

        ypred = model.predict(xtest_scaled)
        scores_KFold[j] = np.sum((ypred - ytest)**2)/np.size(ypred)
        j += 1

    return scores_KFold


def noise_factor(n, factor=0.3):
    return factor*np.random.normal(0, size=n)


if __name__ == '__main__':
    print("Import this file as a package")
