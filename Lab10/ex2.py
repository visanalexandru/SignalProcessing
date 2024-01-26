import numpy as np
import matplotlib.pyplot as plt
import ex1

def graph_line():
    size = 100
    domain = np.linspace(-10, 10, size)
    mean = np.zeros(size)
    covariance = np.zeros((size, size))

    for x in range(size):
        for y in range(size):
            covariance[x, y] = domain[x] * domain[y]

    for x in range(10):
        sample = ex1.sample_multivariate_gaussian(mean, covariance)
        plt.plot(domain, sample)
    plt.title("Lines")
    plt.show()

def graph_brownian():
    size = 100
    domain = np.linspace(0, 10, size)

    mean = np.zeros(size)
    covariance = np.zeros((size, size))

    for x in range(size):
        for y in range(size):
            covariance[x, y] = min(domain[x] , domain[y])

    sample = ex1.sample_multivariate_gaussian(mean, covariance)
    plt.title("Brownian")
    plt.plot(domain, sample)
    plt.show()

def graph_square_exp():
    domain = np.linspace(0,1, 100) 
    alpha = 10 

    mean = np.zeros(len(domain))
    covariance = np.zeros((len(domain), len(domain)))

    for a in range(len(domain)):
        for b in range(len(domain)):
            x = domain[a]
            y = domain[b]

            covariance[a][b] = np.exp(-alpha* np.abs(x-y)**2) 

    samples = ex1.sample_multivariate_gaussian(mean, covariance)

    plt.title("Square exp")
    plt.plot(domain, samples)
    plt.show()

def graph_symmetrical():
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

    samples = ex1.sample_multivariate_gaussian(mean, covariance)

    plt.title("Symmetrical")
    plt.plot(domain,samples)
    plt.show()
    
def graph_period():
    domain = np.linspace(-1,1,100) 
    alpha =  0.5
    beta = 0.8

    mean = np.zeros(len(domain))
    covariance = np.zeros((len(domain), len(domain)))

    for a in range(len(domain)):
        for b in range(len(domain)):
            x = domain[a]
            y = domain[b]

            covariance[a,b] = np.exp(-alpha* (np.sin(beta*np.pi*(x-y))**2))

    samples = ex1.sample_multivariate_gaussian(mean, covariance)

    plt.title("Period")
    plt.plot(domain,samples)
    plt.show()


graph_line()
graph_brownian()
graph_square_exp()
graph_symmetrical()
graph_period()