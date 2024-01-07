import jpeg
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png")
image = np.array(image)

desired_mse = 90

# Decrease the quality until we reach the desired mse.
current_quality = 99
current_mse = 0

while current_mse < desired_mse:
    compressed = jpeg.compress_rgb(image, current_quality)
    decompressed = jpeg.decompress_rgb(compressed)
    current_mse = ((image - decompressed) ** 2).mean()
    print(f"Q: {current_quality}, MSE: {current_mse}")
    current_quality -= 1

plt.imshow(cv2.cvtColor(decompressed, cv2.COLOR_BGR2RGB))
plt.title(f"MSE = {current_mse}")
plt.show()
