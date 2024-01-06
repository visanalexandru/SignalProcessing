import jpeg
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

image = misc.ascent().astype(np.uint8)
print(f"The size of the uncompressed image: {len(image.tobytes())}")

quality = 80
compressed = jpeg.compress_grayscale(image, quality)
print(f"The size of the compressed image: {compressed.size()}")

# Now decompress it to see the difference.
decompressed = jpeg.decompress_grayscale(compressed)
print("MSE: ", ((image - decompressed) ** 2).mean())

fig, axes = plt.subplots(1, 2)
fig.suptitle(f"Q = {quality}")
axes[0].set_title("Original")
axes[0].imshow(image)
axes[1].set_title("Decompressed")
axes[1].imshow(decompressed)
plt.show()
