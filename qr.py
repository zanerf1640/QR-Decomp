import numpy as np
from scipy.linalg import qr as scipy_qr

def random_matrix(m, n):
    return np.random.rand(m, n)

def nearly_dependent_matrix(n,m):
    A = np.random.randn(n, m)
    A[:, -1] = A[:, 0] * 0.999999 + 1e-6
    return A

def hilbert_like(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

def accuracy(A, Q, R):
    return np.linalg.norm(A - Q @ R)

def orthogonality(Q):
    n = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(n))

def time_qr(qr_func, A):
    import time
    start = time.perf_counter()
    Q, R = qr_func(A)
    return Q, R, time.perf_counter() - start

def qr_cgs(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-14:   # skip zero columns
            continue
        Q[:, j] = v / R[j, j]
    return Q, R

def qr_mgs(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] < 1e-14:   
            continue
        Q[:, j] = V[:, j] / R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j] @ V[:, i]
            V[:, i] = V[:, i] - R[j, i] * Q[:, j]
    return Q, R

def qr_householder(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(min(n, m)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        el = np.zeros_like(x)
        el[0] = 1.0
        v = x + np.sign(x[0]) * normx * el
        v = v / np.linalg.norm(v)
        Hk = np.eye(m)
        Hk_sub = np.eye(len(x)) - 2.0 * np.outer(v, v)
        Hk[k:, k:] = Hk_sub
        R = Hk @ R
        Q = Q @ Hk.T
    return Q, R

def qr_givens(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for j in range(n):
        for i in range(m-1, j, -1):
            if abs(R[i, j]) > 1e-14:
                r = np.hypot(R[i-1, j], R[i, j])
                c = R[i-1, j] / r
                s = -R[i, j] / r
                G = np.eye(m)
                G[[i-1, i], [i-1, i]] = c
                G[i, i] = c
                G[i-1, i] = s
                G[i, i-1] = -s
                R = G @ R
                Q = Q @ G.T
    return Q, R

def tall_skinny():
    m, n = 200, 10
    A = np.random.rand(m, n)
    return A

def wide_matrix():
    m, n = 10, 200
    A = np.random.rand(m, n)
    return A

def sparse_banded(n, bandwidth):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            A[i, j] = np.random.rand()
    return A

def qr_numpy(A):
    """Using NumPy's built-in QR decomposition."""
    Q, R = np.linalg.qr(A, mode='reduced')
    return Q, R

def qr_scipy(A):
    """Using SciPy's QR decomposition."""
    Q, R = scipy_qr(A, mode='economic')
    return Q, R


tests = [
("Rand 200x200", random_matrix(200,200)),
("Dep 200x20", nearly_dependent_matrix(200,20)),
("Tall 200x10", tall_skinny()),
("Wide 10x200", wide_matrix()),
("Hilbert 20x20", hilbert_like(20)),
("Sparse 200x200", sparse_banded(200,3))
]
methods = [
("CGS", qr_cgs),
("MGS", qr_mgs),
("HH", qr_householder),
("Givens", qr_givens),
("NumPy", qr_numpy),
("SciPy", qr_scipy)
]

if __name__ == "__main__":
    for test_name, A in tests:
        print(f"Test: {test_name}, shape: {A.shape}")
        for method_name, qr_func in methods:
            Q, R, elapsed = time_qr(qr_func, A)
            acc = accuracy(A, Q, R)
            ortho = orthogonality(Q)
            print(f"  Method: {method_name}, Time: {elapsed:.6f}s, Accuracy: {acc:.2e}, Orthogonality: {ortho:.2e}")
        print()