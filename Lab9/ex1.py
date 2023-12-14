import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7773)

# Generate a random time series that is the sum of three components: Trend, Season, Noise
def generate_time_series(N):
    # Gen N samples
    time = np.linspace(0,3, N)

    trend = (lambda x: 8*x**2 -5*x +3)(time)
    seasonal= (lambda t: 3*np.sin(2*np.pi*10*t) + 12*np.sin(2*np.pi*3*t))(time)
    noise = np.random.normal(size=N)
    return (time, trend, seasonal, noise)

# Use exponential decay to compute a new timeseries.
def exponential_smoothing(series, alpha):
    s = np.zeros(len(series))
    s[0] = series[0]
    for x in range(1, len(series)):
        s[x] = alpha * series[x] + (1-alpha) * s[x-1]
    return s

def test_exponential_smoothing(series, alpha):
    s = exponential_smoothing(series, alpha) 
    error = 0
    cnt = 0
    for x in range(2, len(series)):
        error += (series[x] - s[x-1])**2
        cnt+=1
    return error / cnt

def compute_moving_average(series, q):
    mean = np.mean(series)
    m = len(series) - q + 1
    Y = np.zeros((m, q+1)) 
    noise = np.random.normal(size=len(series))
    print(noise)

    # y[3] = noise[3] + a1*noise[2] + a2*noise[1] + a3*noise[0] + m 
    # ...
    # y[9] = noise[9] + a1 * noise[8] + a2*noise[7] + a3*noise[6] + m
    # y[10] = noise[10] + a1 * noise[9] + a2*noise[8] + a3*noise[7] + m

    # y = Yx 
    # Y = [[noise[3], noise[2], noise[1], noise[0], m]
    #     [[noise[4],  noise[3], noise[2], noise[1], m]
    #     ...
    #     [[noise[10],  noise[9], noise[8], noise[7], m]

    # x = [1, a1, a2, a3, 1]

    for line in range(m):
        for column in range(q):
            Y[line, column] = noise[line+q-column-1] 
        Y[line, column+1] = mean 

    print(Y)


t , trend, seasonal, noise = generate_time_series(4000) 
y = trend+seasonal+noise

# Plotting the time series.
fig, axs = plt.subplots(4, figsize=(10,10))
fig.suptitle("Univariate time series")
fig.tight_layout(pad=3)
axs[0].plot(t, y)
axs[0].set_title("Observed")
axs[1].plot(t, trend)
axs[1].set_title("Trend")
axs[2].plot(t, seasonal)
axs[2].set_title("Seasonal")
axs[3].plot(t, noise)
axs[3].set_title("Noise")

# Plotting the time series passed through exponential decay.
fig, axs = plt.subplots(1, figsize=(10,10))
exp_decay = exponential_smoothing(y[:200], 0.4)
axs.plot(t[:200], exp_decay)
axs.plot(t[:200], y[:200])

# Finding a good alpha.
alpha_values = np.linspace(0.2, 1, 100)

errors = []
errors_alpha = []

for alpha in alpha_values:
    error = test_exponential_smoothing(y, alpha)
    errors.append(error)
    errors_alpha.append(alpha)

min_error = np.argmin(errors)
best_alpha = errors_alpha[min_error]

fig, axs = plt.subplots(1, figsize=(10,10))
axs.plot(errors_alpha, errors)
axs.stem(best_alpha, errors[min_error])
plt.show()


# Computing the moving average model.

compute_moving_average([1,2,3,4, 5, 6, 7],3)
