from numpy import *
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import load_digits,load_boston
#from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation  import train_test_split
 


def cvxOptimization(X, y,C):
    N = X.shape[0]
    d = X.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * X
    K = np.dot(K, K.T)
    K = K.astype(np.double)
#    print(K.shape)
    P = matrix(K)
#    P=P.astype(double)
#    print(P.shape)
#    print(len(P))
#    print(P.size)
    #size(P)
    y = y.astype(np.double)
    tmp1 = np.diag(np.ones(N) * -1)
    tmp2 = np.identity(N)
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(N)
    tmp2 = np.ones(N) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    q = matrix(-np.ones((N, 1)))
    #G = matrix(-np.eye(N))
    #h = matrix(np.zeros(N))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    #print("k")
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def SVM_Dual(X,y,C):
    alphas = cvxOptimization(X,y,C)
    W = np.sum(alphas * y[:, None] * X, axis = 0)
    # get bias
    cond = (alphas > 0).reshape(-1)
    b = y[cond] - np.dot(X[cond], W)
    #bias = b[0]
    print("b is",b)
    bias = b.mean()
    print("W is:",W)
    print("Bias is:", bias)
    return W,bias

def predict(X,W,b):
    y = np.dot(X,W) + b
    target = np.array([1 if(i>0) else -1 for i in y])
    print(target)
    return target

def calculateError(real,predicted):
    error = np.sum(((real - predicted) != 0))*1.0/ real.shape[0]
    print("Real is:",real)
    print("predicted is:",predicted)
    print("Difference is:", np.sum(((real-predicted) != 0)))
    print(real.shape[0])
    print(error)
    return error

def main():
   # if(len(sys.argv)>1):
    #    filename = sys.argv[1]
    #    num_splits = int(sys.argv[2])
    #    C_values = [int(i) for i in sys.argv[3]]
    #else:
    filename = "MNIST-13.csv"
    num_splits = 10
    
    C_values = [1e-5,1e-6,1e-7,1e-8,1e-9]
    #Load data from file.
    if(filename == 'boston.csv'):
        print("Taking boston data")
        boston = load_boston()
        HomeVal50 = {}
        HomeVal50["data"] = boston.data
        HomeVal50["target"] = np.array([1 if(y>np.median(boston.target)) else 0 for y in boston.target])
        data  = HomeVal50["data"]
        target= HomeVal50["target"]
        
    else:
        print("Taking MNIST-13 data")
        df = pd.read_csv(filename)
        Data = np.array(df)
        y = Data[:,0]
        data = Data[:,1:]
        target = np.array([1 if(i == 1) else -1 for i in y])
    
    
    #train_errors = np.zeros([num_splits,len(train_percent)])
#     train_percent[percent_i]/100
    test_errors = np.zeros([num_splits,len(C_values)])
    for C in range(len(C_values)):
        for split in range(num_splits):
            X_train,X_test,y_train,y_test = train_test_split(data, target, test_size=0.20, stratify=target)
            W,b = SVM_Dual(X_train,y_train,C_values[C])
            test_errors[split][C] = calculateError(y_test,predict(X_test,W,b))
            print(test_errors)
    #print(test_errors)
    test_errors = np.array(test_errors)
    mean_test_errors = np.mean(test_errors,axis = 0)
    std_test_errors = np.std(test_errors,axis=0)
    print(test_errors)
    print("Mean of testing errors for respective training percetages are:",mean_test_errors)
    print("Standard deviation of testing errors for respective training percetages are:",std_test_errors)
    #errorbar(train_percent, mean_test_errors, yerr=std_test_errors, capthick=4)
#     xlabel('% of training data')
#     ylabel('Mean test errors')
#     title('Mean test errors variation')
#     show()
    return test_errors

if __name__ == '__main__':
    test_errors = main()



