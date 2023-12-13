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


# 0 1 2 3 4 5 6
#   0 1 2 3 4 5 6
# Computes the autocorrelation vector for the given vector.
def autocorrelation(x):
    # See: https://www.mathworks.com/help/econ/autocorr.html

    x_mean = np.mean(x)
    autocor = np.zeros(len(x))

    for offset in range(len(x-1)):
        for i in range(0, len(x)-offset):
            autocor[offset] += (x[i] - x_mean) * (x[i+offset] - x_mean)

    autocor = autocor/autocor[0]
    return autocor

# Returns the coefficients of an AR model trained on the dataset supplied.
def fit_ar(dataset, p):
    lines = len(dataset)-p
    Y = np.zeros(shape=(lines, p))
    
    for y in range(lines):
        for x in range(p):
            Y[y,x] = dataset[p+y-x-1] 
    y = dataset[p:].T
    coeffs = np.matmul(np.linalg.pinv(Y), y) 
    return coeffs 

# Predicts the next value in the dataset using an ar model.
def predict_ar(dataset, coeffs, num_predictions):
    predicted = []
    for x in range(num_predictions):
        # Get the last len(coeffs) values.
        if x >= len(coeffs):
            window = np.array(predicted[-len(coeffs):])
        else:
            a = dataset[-len(coeffs)+x:]
            b = np.array(predicted) 
            window = np.concatenate((a,b))
        next = np.convolve(window, coeffs, mode="valid")
        predicted.append(*next)
    return np.array(predicted)

# Generating a new time series.
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

# Computing the autocorrelation of the time series.
autocorr = autocorrelation(y)
fig, axs = plt.subplots(1, figsize=(10,10))
axs.plot(autocorr)
axs.set_title("Autocorrelation vector")

# Fitting an AR model.
m = int(0.8 * len(y))
p = 350 
train = y[:m] 
coeffs = fit_ar(train, p)

# Predicting the next values of the train set.
predictions = predict_ar(train, coeffs, len(y)-m)
fig, axs = plt.subplots(1, figsize=(10,10))
axs.plot(t[m:],predictions, color="red", label="predicted")
axs.plot(t[m:], y[m:], color="green", label="actual")
axs.plot(t[:m], y[:m], color="blue", label="train set")
axs.legend()
plt.show()

# Cross validation.
# ┌────────────────────────────────┐
# │                                │
# │         Original data          │
# │                                │
# └────────────────────────────────┘
#  ┌─────────┬─┐
#  │         │ │
#  │  Train  │?│
#  │         │ │
#  └─────────┴─┘
#            ┌─────────┬─┐
#            │         │ │
#            │  Train  │?│
#            │         │ │
#            └─────────┴─┘
#                      ┌─────────┬─┐
#                      │         │ │
#                      │  Train  │?│
#                      │         │ │
#                      └─────────┴─┘
# This method tests the performance of the ar model with the given params.
# It splits the data into folds like the drawing above. 
# For each fold, compute the test error by making just one prediction.
# Compute all the test errors and return the average.
# Use MSE as a performance metric.
def get_ar_error(dataset, m, p):
    errors = []
    # Generate validation sets of size m.
    for y in range(0, len(dataset)-m-1,m):
        train_set = dataset[y: y+m]
        coeffs = fit_ar(train_set, p)
        test = dataset[y+m+1]
        prediction = predict_ar(train_set, coeffs, 1)[0]
        errors.append((prediction-test)**2)

    return np.mean(np.array(errors))
    
m_values = [20, 40, 80, 160, 320, 640]
p_values = [2, 4, 8, 16, 32, 64, 128, 256]
smallest_error = np.inf
m_sol, p_sol = np.nan, np.nan

for m in m_values:
    for p in p_values:
        if p >=m:
            continue
        error = get_ar_error(y, m, p)
        print(f"m={m}, p={p}, error={error}")
        if error<smallest_error:
            m_sol = m
            p_sol = p 
            smallest_error = error

print(f"Best: m={m_sol}, p={p_sol}, error={smallest_error}")