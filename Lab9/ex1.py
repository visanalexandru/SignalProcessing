import numpy as np
import matplotlib.pyplot as plt
import scipy

np.random.seed(71173)

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
    for x in range(1, len(series)):
        error += (series[x] - s[x-1])**2
        cnt+=1
    return error / cnt

# Find the best alpha.
def optimise_exponential_smoothing(series):
    alpha_values = np.linspace(0.2, 1, 100)

    # Finding a good alpha.
    errors = []
    errors_alpha = []

    for alpha in alpha_values:
        error = test_exponential_smoothing(series, alpha)
        errors.append(error)
        errors_alpha.append(alpha)

    min_error = np.argmin(errors)
    best_alpha = errors_alpha[min_error]

    fig, axs = plt.subplots(1, figsize=(10,10))
    fig.suptitle("Optimising the alpha value")
    axs.plot(errors_alpha, errors)
    axs.stem(best_alpha, errors[min_error])
    return best_alpha


def compute_moving_average(series, q, noise):
    mean = np.mean(series)
    # Example for p = 4
    # y[4] = e4 + a1*e3 + a2*e2 + a3*e1 + a4*e0 + mean
    # y[5] = e5 + a1*e4 + a2*e3 + a3*e2 + a4*e1 + mean
    # y[6] = e6 + a1*e5 + a2*e4 + a3*e3 + a4*e2 + mean
    # ...
    # We can translate this to: 
    # y[4] - e4 - mean = a1*e3 + a2*e2 + a3*e1 + a4*e0
    # y[5] - e5 - mean = a1*e4 + a2*e3 + a3*e2 + a4*e1
    # y[6] - e6 - mean = a1*e5 + a2*e4 + a3*e3 + a4*e2
    # ...

    num_lines = len(series) - q 
    Y = np.zeros((num_lines, q))
    x = np.zeros(num_lines)

    for line in range(num_lines):
        for column in range(q):
            Y[line, column] = noise[q + line -1 - column]
        
        x[line] = series[q + line] - noise[q+line] - mean 
    
    # We must solve: x = Y * a to get our coefficients.

    return np.matmul(np.linalg.pinv(Y), x) 

# Use the ma model to make predictions.
def predict_moving_average(mean, noise, ma):
    q = len(ma)
    result = []
    for x in range(q, len(noise)):
        last = noise[x-q:x]
        value = np.convolve(last, ma, mode="valid") + mean + noise[x]
        result.append(value)
    return np.array(result)

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


# Finding the best parameter for exponential smoothing.
best_alpha = optimise_exponential_smoothing(y)
print(f"The best alpha is: {best_alpha}")

# Plotting the time series passed through exponential smoothing.
fig, axs = plt.subplots(2, figsize=(10,10))
exp_default= exponential_smoothing(y[:200], 0.8)
exp_best= exponential_smoothing(y[:200], best_alpha)

axs[0].set_title("Exponential smoothing using alpha = 0.8")
axs[1].set_title(f"Exponential smoothing using alpha = {best_alpha}")
axs[0].plot(t[:200], exp_default, label="smooth")
axs[0].plot(t[:200], y[:200], label="original")
axs[1].plot(t[:200], exp_best, label="smooth")
axs[1].plot(t[:200], y[:200], label="original")
fig.legend()



# Computing the MA model.
noise = np.random.normal(size=len(y))
q = 800 
train_size = 2000
train_y = y[:train_size]
train_noise = noise[:train_size]
test_noise = noise[train_size:]
mean = np.mean(train_y)

ma = compute_moving_average(train_y, q, train_noise) 
result = predict_moving_average(mean, test_noise, ma)

fig, axs = plt.subplots(2, figsize=(10,10))
axs[0].plot(t[train_size+q:], y[train_size+q:])
axs[1].plot(t[train_size+q:], result)
plt.show()
