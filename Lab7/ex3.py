from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import ex2


# Loading image and generating noisy image like in ex2.
X = misc.face(gray=True)
height, width = X.shape
pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
Y_noisy = np.fft.fft2(X_noisy)

# Denoising the image with the optimal param.
optimal = 0.05
Y_cleared = ex2.filter_freqs(Y_noisy, optimal,  width, height)
X_cleared = np.fft.ifft2(Y_cleared).real

fig, axs = plt.subplots(2, figsize=(10,10))

axs[0].imshow(X_cleared, cmap = plt.cm.gray)
axs[1].imshow(X_cleared-noise, cmap = plt.cm.gray)
plt.show()


print(ex2.snr(X_cleared, noise))
print(ex2.snr(X_cleared-noise, noise))