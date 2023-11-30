from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Compute the snr of the image,
def snr(image):
    a = np.mean(image)
    b = np.std(image)
    return a/b


def filter_freqs(Y, ratio, width, height):
    Y_cleared = Y.copy()
    Y_cleared[int(ratio*height):int((1-ratio)*height), :] = 0
    Y_cleared[:, int(ratio*width):int((1-ratio)*width)] = 0
    return Y_cleared


if __name__ == "__main__":
    fig, axs = plt.subplots(5,2, figsize=(10,10))
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
    
    # Removing high frequencies from the original image.
    for i, freq_cutoff in enumerate([0.4, 0.1, 0.05, 0.02]):
        Y_cleared = filter_freqs(Y, freq_cutoff, width, height) 
        X_cleared = np.fft.ifft2(Y_cleared).real

        title = f"SNR = {snr(X_cleared)}"
        axs[1+i,0].set_title(title)
        axs[1+i,0].imshow(X_cleared, cmap=plt.cm.gray)
        axs[1+i,1].set_title(f"Cut FFT 2d, ratio = {freq_cutoff}")
        axs[1+i,1].imshow(20*np.log10(abs(Y_cleared)) )
    plt.show()

