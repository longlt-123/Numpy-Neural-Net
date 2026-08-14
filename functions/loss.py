import numpy as np

def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False, multioutput: str = 'uniform_average'):
    if derivative:
        dA = 2 * (y_pred - y_true)
        if multioutput == 'raw_values':
            return dA / y_true.shape[1]
        elif multioutput == 'sum':
            return dA / y_true.shape[1]
        elif multioutput == 'uniform_average':
            return dA / (y_true.shape[0] * y_true.shape[1])

    loss_matrix = np.power(y_pred - y_true, 2)

    if multioutput == 'raw_values':
        return np.mean(loss_matrix, axis=-1)
    elif multioutput == 'sum':
        return np.sum(np.mean(loss_matrix, axis=-1))
    elif multioutput == 'uniform_average':
        return np.mean(np.mean(loss_matrix, axis=-1))
        
    else:
        raise ValueError("Invalid 'multioutput' option.")

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False, multioutput: str = 'uniform_average', epsilon: float = 1e-10):
    y_pred_clipped = np.clip(y_pred, epsilon, 1-epsilon)

    if derivative:
        dA = - (y_true / y_pred_clipped) + (1 - y_true) / (1 - y_pred_clipped)
        
        if multioutput == 'raw_values':
            return dA / y_true.shape[1]
        elif multioutput == 'sum':
            return dA / y_true.shape[1]
        elif multioutput == 'uniform_average':
            return dA / (y_true.shape[0] * y_true.shape[1])

    loss_matrix = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    if multioutput == 'raw_values':
        return np.mean(loss_matrix, axis=-1) 
    elif multioutput == 'sum':
        return np.sum(np.mean(loss_matrix, axis=-1))
    elif multioutput == 'uniform_average':
        return np.mean(np.mean(loss_matrix, axis=-1))
        
    else:
        raise ValueError("Invalid 'multioutput' option.")


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False, multioutput: str = 'uniform_average', epsilon: float = 1e-10):
    y_pred_clipped = np.clip(y_pred, epsilon, 1-epsilon)

    if derivative:
        dA = -y_true / y_pred_clipped

        if multioutput == 'raw_values':
            return dA
        elif multioutput == 'sum':
            return dA
        elif multioutput == 'uniform_average':
            return dA / y_true.shape[0]

    if multioutput == 'raw_values':
        return -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
    elif multioutput == 'sum':
        return -np.sum(y_true * np.log(y_pred_clipped))
    elif multioutput == 'uniform_average':
        return -np.average(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
    else:
        raise ValueError("Invalid 'multioutput' option.")