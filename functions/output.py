import numpy as np

def softmax(x: np.ndarray, axis=-1, derivative: bool = False) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    s = e / np.sum(e, axis=axis, keepdims=True)

    if derivative:
        return np.einsum('ij,jk->ijk', s, np.eye(s.shape[axis]), optimize=True) - np.einsum('ij,ik->ijk', s, s, optimize=True)

    return s