import numpy as np
import matplotlib.pyplot as plt


def sample_gaussian(mean, variance, size):
    stdev = np.sqrt(variance)
    return np.random.normal(mean, stdev, size=size)

def sample_multivariate_gaussian(mean_vector, variance_matrix):
    dim = len(mean_vector)
    assert variance_matrix.shape == (dim, dim)

    u, s, _ = np.linalg.svd(variance_matrix)
    s = np.diag(s)

    n = np.random.normal(0, 1, size=dim)
    x = np.matmul(u, np.sqrt(s))
    x = np.matmul(x, n)
    return x + mean_vector

def graph_univariate_gauss():
    samples = sample_gaussian(0, 4, 10000)
    plt.title("Univariate normal distribution")
    plt.hist(samples, bins=100, density=True)
    plt.show()

def graph_multivariate_gauss():
    mean_vector = np.array([0,0])
    covariance_matrix = np.array([[1, 3/5], [3/5, 2]])
    samples = np.array([sample_multivariate_gaussian(mean_vector, covariance_matrix) for _ in range(1000)])

    plt.title("2d normal distribution")
    plt.axis('equal')
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()

if __name__ =="__main__":
    graph_univariate_gauss()
    graph_multivariate_gauss()