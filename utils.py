import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import subprocess
import pickle
from scipy.sparse import csr_matrix


def is_doubly_stochastic(P):
    eps = 10e-2
    m, n = P.shape
    for i in range(m):
        sum = 0
        sum2 = 0
        for j in range(n):
            sum += P[i][j]
            sum2 += P[j][i]
        if (np.abs(sum -1) > eps) or (np.abs(sum2 -1) > eps):
            return False
    return True


class RewNumPulls:
    def __init__(self, n, rewards_array, pulls_array):
            self.rewards = rewards_array
            self.pulls = pulls_array


def sparsify(P):
    rows = []
    columns = []
    values = []
    it = np.nditer(P, flags=['multi_index'],op_flags=['readonly'])

    while not it.finished:
        if it[0] != 0:
            rows.append(it.multi_index[0])
            columns.append(it.multi_index[1])
            values.append(it[0])
        it.iternext()
    return csr_matrix((values,(rows, columns)))


class AccelMix:
    def __init__(self, P, lambda_2, C, sparse=True):
        w = 0.5
        old_w = 0
        new_w = None
        for i in range(C):
            new_w = 2*w/lambda_2 - old_w
            old_w, w = w, new_w
            if i == 0:
                old_w *= 2
        self.final_w = new_w
        self.lambda_2 = lambda_2
        self.C = C
        if not sparse:
            self.P = P
        else:
            self.P = sparsify(P)

    def mix(self, y):
        y /= 2.0
        old_y = 0
        old_y, y = y, (2/self.lambda_2)*self.P.dot(y) - old_y
        old_y *= 2

        for i in range(1, self.C):
            old_y, y = y, (2/self.lambda_2)*self.P.dot(y) - old_y

        y /= self.final_w
        return y

class UnaccelMix:
    def __init__(self, P, lambda_2, C=1, sparse=False):
        self.C = C
        if C != 1:
            self.P = np.linalg.matrix_power(P, C)
        else:
            self.P = P
        if C == 1 or sparse:
            self.P = sparsify(self.P)

    def mix(self, y):
        return self.P.dot(y)


def accel_mix(y, C, P, lambda_2):
    w = 0.5
    old_w = 0
    y /= 2.0
    old_y = 0
    new_w = None
    for i in range(C):
        new_w = 2*w/lambda_2 - old_w
        #old_y, y = y, (w/new_w)*(2/lambda_2)*np.dot(P, y) - old_w/new_w*old_y
        old_y, y = y, (2/lambda_2)*np.dot(P, y) - old_y

        old_w, w = w, new_w
        if i == 0:
            old_y *= 2
            old_w *= 2

    y /= new_w
    return y


def unaccel_mix(y, C, P, lambda_2):
    if C == 1:
        return np.dot(P, y)
    else:
        return np.dot(np.linalg.matrix_power(P, C), y)


def compute_constants(P, n):
    # Returns epsilon_c and lambda_2
    lambda_, eigenvectors = la.eig(P)
    lambda_ = -np.array(sorted(-lambda_))
    lambda_ = lambda_.astype(float)
    eigenvectors = eigenvectors.astype(float)
    # epsilon_n = math.sqrt(n)*sum(lambda_[1:]/(1-lambda_)[1:])

    b = np.zeros((n, n, n))
    t = np.zeros((n, n, n))
    for p in range(n):
        for j in range(n):
            try:
                b[p][j] = np.fabs(np.dot(eigenvectors[p], eigenvectors[j])
                                   * eigenvectors[p]*eigenvectors[j])
            except:
                #print(np.dot(eigenvectors[p], eigenvectors[j]) * eigenvectors[p]*eigenvectors[j])
                raise
    aux = np.fabs(np.outer(lambda_[1:], lambda_))
    aux /= 1-aux
    aux = aux.reshape(aux.shape + (1,))
    epsilon_c = np.sum(np.multiply(b[1:,:,:], aux), axis=(0,1))
    epsilon_c *= n

    return epsilon_c, lambda_[1] # Eigenvalues are positive, so \lambda_2 is lambda_[1]

def compute_P(n, type, args=None):
    if type == 'cycle':
        A = np.zeros((n, n), dtype=float)
        D_sqrt_inv = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(1,2):
                A[i][(i+j)%n] = 1
                A[i][(i-j)%n] = 1
            A[i][i] = 0
        delta = [A[0][j] for j in range(n)].count(1)
        for i in range(n):
            D_sqrt_inv[i][i] = 1/np.sqrt(delta)
        Lap = np.eye(n, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        P = np.eye(n, dtype=float) - delta/(delta+1)*Lap
    elif type == 'grid':
        aux = int(round(math.sqrt(n)))
        if n != aux * aux:
            raise RuntimeError("With type {} the number of nodes n must be a square number. It was {}".format(type, n))
        A = np.zeros((n, n), dtype=float)
        for x in range(n):
            i = x//aux
            j = x % aux
            if (i+1) < aux:
                A[x][aux*(i+1)+j] = 1
            if (i-1) >=0:
                A[x][aux*(i-1)+j] = 1
            if (j+1) < aux:
                A[x][aux*i + j+1] = 1
            if (j-1) >=0:
                A[x][aux*(i) + j-1] = 1
        delta = np.array([[A[i][j] for j in range(n)].count(1) for i in range(n)])
        D_sqrt_inv = np.diag(1/np.sqrt(delta))

        Lap = np.eye(n, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        # Note that since the grid is not a regular graph, we have to use a different formula for P here. See Duchi et al 2012
        P = np.eye(n, dtype=float) - 1/(delta.max()+1)*np.dot(np.dot(D_sqrt_inv, Lap), D_sqrt_inv)


    if not is_doubly_stochastic(P):
        raise RuntimeError("P is not double stochastic")
    return P


def plot(regrets, n, T, type_P, sigma, K):
    for regret, label in regrets:
        plt.plot(regret, label=label)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Iterations')
    plt.ylabel("Regret")

    plt.legend(loc='upper left', frameon=True, framealpha=1)
    plt.title('')

    filename = 'results_n{}_t{}_{}.pdf'.format(n, T, type_P)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    #subprocess.Popen(["evince {}".format(filename)],shell=True)


def read_regret(filename):
    fileObject = open(filename,'rb')
    return pickle.load(fileObject)
