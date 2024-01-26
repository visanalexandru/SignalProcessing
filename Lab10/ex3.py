from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import ex1

def get_trend(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    return trend.squeeze()


# Use square exp kernel.
def compute_covariance(domain1, domain2):
    alpha = 0.0002

    n1 = len(domain1)
    n2 = len(domain2)

    covariance = np.zeros(shape=(n1, n2))

    for a in range(n1):
        for b in range(n2):
            x = domain1[a]
            y = domain2[b]

            covariance[a][b] = 100*np.exp(-alpha* np.abs(x-y)**2) 
    return covariance


co2 = fetch_openml(data_id=41187, as_frame=True)

co2_data = co2.frame
co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
co2_data = co2_data[["date", "co2"]].set_index("date")

# Take the monthly average.
co2_data = co2_data.resample("M").mean().dropna(axis="index", how="any")

# Removing the trend.
samples = np.array(co2_data["co2"])
domain = np.array(range(0, len(samples)))/len(samples)
trend = get_trend(domain, samples)
print(trend.shape)

fig, axes = plt.subplots(2, 1)
fig.suptitle("Eliminating the trend")
axes[0].plot(domain, samples)
axes[0].plot(domain, trend)
axes[1].plot(domain, samples-trend)
plt.show()
samples = samples - trend

# Train test split
test_dim = 100
train = samples[:-test_dim]
train_domain = domain[:-test_dim]

test = samples[-test_dim:]
test_domain = domain[-test_dim:]

# Gaussian regression. 
test_train_cov = compute_covariance(test_domain, train_domain)
train_test_cov = compute_covariance(train_domain, test_domain)
train_train_cov = compute_covariance(train_domain, train_domain)
test_test_cov = compute_covariance(test_domain, test_domain)

# m = mean(test) + Cov(test, train) * Cov(train_train)^(-1) * (train - mean(train))
m = np.matmul(test_train_cov ,np.linalg.pinv(train_train_cov))
m = np.matmul(m, (train - np.mean(train_domain)))
m += np.mean(test_domain)

cov = test_test_cov - np.matmul(np.matmul(test_train_cov, np.linalg.pinv(train_train_cov)), train_test_cov)

for x in range(20):
    predictions = ex1.sample_multivariate_gaussian(m, cov)
    plt.plot(test_domain, predictions, alpha=0.2, c="b")

plt.plot(train_domain, train)
#plt.plot(test_domain, test)
plt.plot(test_domain, m)
plt.show()