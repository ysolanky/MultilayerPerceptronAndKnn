# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.

import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return np.sqrt(sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return sum(abs(x1 - x2))


def identity(x, derivative=False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # definition of function taken from # https://en.wikipedia.org/wiki/Activation_function
    if derivative:
        return np.ones((x.shape[0], x.shape[1]))
    else:
        return x


def sigmoid(x, derivative=False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # definition of function taken from # https://en.wikipedia.org/wiki/Activation_function
    # To solve overflow warning I used np.clip https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
    x = np.clip(x, -709.78, 709.78)
    # End of code taken from stackoverflow

    temp = 1 / (1 + np.exp(-x))

    if derivative:
        return temp * (1 - temp)
    else:
        return temp


def tanh(x, derivative=False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # definition of function taken from # https://en.wikipedia.org/wiki/Activation_function
    # To solve overflow warning I used np.clip https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
    x = np.clip(x, -709.78, 709.78)
    # End of code taken from stackoverflow

    a = (np.exp(x) - np.exp(-x))
    b = (np.exp(x) + np.exp(-x))

    temp = a / b

    if derivative:
        return 1 - (temp) ** 2
    else:
        return temp


def relu(x, derivative=False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # To avoid OverflowError: Python int too large to convert to C long, I added the following line of code
    # from https://stackoverflow.com/questions/38314118/overflowerror-python-int-too-large-to-convert-to-c-long-on-windows-but-not-ma
    x = np.array(x, dtype=np.int32)
    # End of code from stack overflow

    relu = lambda t: max(t, 0)

    # Code taken from # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    vector = np.vectorize(relu)
    # End of code taken from stack overflow

    # I created the function below taking inspiration from the function above.
    relu_d = lambda t: 0 if t <= 0 else 1

    vector_d = np.vectorize(relu_d)

    if derivative:
        return vector_d(x)
    else:
        return vector(x)


def softmax(x, derivative=False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    # definition of cross entropy loss taken from https://en.wikipedia.org/wiki/Cross_entropy
    np.seterr(divide='ignore')
    np.seterr(invalid="ignore")
    # ignoring the warnings - https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log

    return -np.sum(y * np.log(p))


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    temp = np.zeros((len(y), len(np.unique(y))))

    columns = {}

    count = 0
    for i in np.unique(y):
        columns[i] = count
        count += 1

    count = 0
    for i in y:
        temp[count][columns[i]] = 1
        count += 1
    return temp
