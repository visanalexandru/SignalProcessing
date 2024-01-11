import numpy as np
import matplotlib.pyplot as plt

def sample_multivariate_gaussian(mean_vector, covariance_matrix):
    n = np.random.normal(size=len(mean_vector))
    u, s, vh = np.linalg.svd(covariance_matrix)

    s = np.diag(np.sqrt(s))
    result = np.matmul(u,s) 
    result = np.matmul(result, n) 
    result += mean_vector

    return result


def graph_univariate_gauss():
    mean = 3
    stdev = 1
    num_samples = 1000
    samples = np.random.normal(mean, stdev, size=num_samples) 
    plt.hist(samples, bins=50)
    plt.show()

graph_univariate_gauss()

def graph_multivariate_gauss():
    mean_vector = [0,0]
    mat = np.array([[1, 3/5], [3/5, 2]]) 

    num_samples = 1000
    samples = [sample_multivariate_gaussian(mean_vector, mat) for x in range(num_samples)]

    samples_y = [sample[0] for sample in samples]
    samples_x = [sample[1] for sample in samples]

    plt.axis('equal')
    plt.scatter(samples_x, samples_y)
    plt.show()


graph_multivariate_gauss()


def graph_square_exp():
    domain = np.linspace(0,1, 100) 
    alpha = 0.01

    mean = np.zeros(len(domain))
    covariance = np.zeros((len(domain), len(domain)))

    for a in range(len(domain)):
        for b in range(len(domain)):
            x = domain[a]
            y = domain[b]

            covariance[a][b] = np.exp(-alpha* np.abs(x-y)) 

    samples = sample_multivariate_gaussian(mean, covariance)

    plt.plot(samples)
    plt.show()

def graph_simmetrical():
    domain = np.linspace(-5,5,100) 
    alpha = 1 

    mean = np.zeros(len(domain))
    covariance = np.zeros((len(domain), len(domain)))

    for a in range(len(domain)):
        for b in range(len(domain)):
            x = domain[a]
            y = domain[b]

            diff = abs(x-y)
            sum = abs(x+y)

            m = min(diff, sum)
            covariance[a][b] = np.exp(-alpha * m**2) 

    samples = sample_multivariate_gaussian(mean, covariance)

    plt.plot(domain,samples.T)
    plt.show()




graph_square_exp()
graph_simmetrical()
    
