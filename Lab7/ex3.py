from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import ex2

if __name__ == "__main__":
    fig, axs = plt.subplots(6,2, figsize=(10,10))
    fig.tight_layout(h_pad=2)

    # Loading the racoon image.
    X = misc.face(gray=True)
    height ,width = X.shape 
    axs[0,0].set_title("Original image")
    axs[0,0].imshow(X, cmap=plt.cm.gray)

    # Computing the fft2d of the original image.
    Y = np.fft.fft2(X)
    axs[0,1].set_title("FFT 2d of the original image")
    axs[0,1].imshow(20*np.log10(abs(Y)))

    # Generating the noisy image. 
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
    X_noisy = X + noise
    axs[1,0].set_title(f"Noisy image, SNR = {ex2.snr(X_noisy)}")
    axs[1,0].imshow(X_noisy, cmap=plt.cm.gray)

    # Generating the fft2d of the noisy image.
    Y_noisy = np.fft.fft2(X_noisy)
    axs[1,1].set_title("FFT 2d of the noisy image")
    axs[1,1].imshow(20*np.log10(abs(Y_noisy)))

    # Removing high frequencies in the noisy image.
    for i, freq_cutoff in enumerate([0.4, 0.1, 0.05, 0.02]):
        Y_cleared = ex2.filter_freqs(Y_noisy, freq_cutoff, width, height) 
        X_cleared = np.fft.ifft2(Y_cleared).real

        title = f"SNR = {ex2.snr(X_cleared)}"
        if i == 2:
            title += " (optimal)"
        axs[2+i,0].set_title(title)
        axs[2+i,0].imshow(X_cleared, cmap=plt.cm.gray)
        axs[2+i,1].set_title(f"Cut FFT 2d, ratio = {freq_cutoff}")
        axs[2+i,1].imshow(20*np.log10(abs(Y_cleared)) )
    plt.show()

