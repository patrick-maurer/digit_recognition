import numpy as np


def logistic(x):
    """
    standard logistic function

    :param x: array
    :return: logistic function applied to array
    """

    mask = x > 0
    y = np.full(shape=x.shape, fill_value=np.nan)
    y[mask] = 1 / (1 + np.exp(-(x[mask])))
    y[~mask] = np.exp(x[~mask]) / (np.exp(x[~mask]) + 1)
    return y


def softmax(x):
    """
    softmax

    :param x: 2d array with one row per sample
    :return: 2d array with the same shape as the input, with softmax applied to each row-vector
    """

    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


def one_hot(x):
    """
    one-hot encoding of an array

    :param x: 1d array where element i gives the true label for sample i
    :return: tuple of (onehot, classes) where:
             - onehot is a (n x k) array where n = len(x), k = len(np.unique(x)) and
               element (i,j) = 1 if x[i] == np.unique(x)[j], 0 otherwise
             - classes is a 1d array of classes corresponding to the columns of onehot
    """

    classes, inverse = np.unique(x, return_inverse=True)
    onehot = np.eye(classes.shape[0], dtype="int64")[inverse]
    return (onehot, classes)


def cross_entropy(Yhat, Y):
    """
    row-wise cross entropy

    :param Yhat: (n x k) array where (i,j) gives the predicted probability of class j for sample i
    :param Y: either:
              1) (n x k) array where (i,j) gives the true class j for sample i or
              2) a 1-D array where element i gives the index of the true class for sample i
    :return: 1-D array with n elements, where element i gives the cross entropy for the ith sample
    """

    if Y.ndim == 1:
        ce = -np.log(Yhat[np.arange(len(Y)), Y])
    else:
        ce = -np.sum(Y * np.log(Yhat), axis=1)

    return ce
