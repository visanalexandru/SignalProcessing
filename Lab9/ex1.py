import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(71173)

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


def compute_moving_average(series, q):
    mean = np.mean(series)
    m = len(series) - q + 1


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

plt.show()
