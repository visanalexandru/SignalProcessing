import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(x):
    result = np.zeros(len(x))

    for l in range(len(x)):
        a = x
        b = np.append(x[l:] , np.zeros(l))
        result[l] = np.dot(a, b)
    return result


# Build an autoregresive model for the given time series, with dim = p.
def build_ar_y(past, p):
    # Example for p = 2:
    # y[2] = x1 * y[0] + x2 * y[1]

    m = len(past) - p
    Y = np.zeros((m, p))


    for y in range(m):
        for x in range(p):
            Y[y,x] = past[p+y-x-1]

    print(Y)
    y = past[p:].T 

    x = np.linalg.inv(np.matmul(Y.T, Y)) 
    x = np.matmul(np.matmul(x, Y.T), y)

    return x 


# a)
np.random.seed(100)
N = 2000
space = np.linspace(0, 1, N)

# Generate the trend of the time series.
trend_f = lambda x : 4*x**2 + 3*x + 10
trend = trend_f(space)

# Generate the seazon.
f1 = 10
f2 = 5 
seazon_f = lambda t : 2*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
seazon = seazon_f(space)

# Generate the noise
noise = np.random.normal(size=len(space))

series = trend + seazon + noise

fig, axs = plt.subplots(4, figsize=(10,10))
axs[0].set_title("Trend")
axs[0].plot(space, trend)
axs[1].set_title("Seasonal")
axs[1].plot(space, seazon)
axs[2].set_title("Residual")
axs[2].plot(space, noise)
axs[3].set_title("Observed")
axs[3].plot(space, series)
plt.show()


# b)
# Computing the autocorrelation. 
c = autocorrelation(series) 
fig, axs = plt.subplots(1, figsize=(10,10))

axs.plot(c)
axs.set_title("Autocorrelation vector")

plt.plot(c)
plt.show()

# c)
p = 200
m = 900 

past = series[:m]
X = build_ar_y(past, p)

for x in range(len(series) - m):
    window = past[-p:]
    print(window, X)
    prediction = np.convolve(X, window, mode="valid") 
    past = np.append(past, prediction[0])

plt.plot(past[m:])
plt.plot(series[m:])
plt.show()
