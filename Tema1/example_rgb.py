import jpeg
import matplotlib.pyplot as plt
from scipy import misc

image = misc.face()
print(f"The size of the uncompressed image: {len(image.tobytes())}")

compressed = jpeg.compress_rgb(image)
print(f"The size of the compressed image: {compressed.size()}")

# Now decompress it to see the difference.
decompressed = jpeg.decompress_rgb(compressed)
print("MSE: ", ((image - decompressed) ** 2).mean())

fig, axes = plt.subplots(1, 2)
axes[0].set_title("Original")
axes[0].imshow(image)
axes[1].set_title("Decompressed")
axes[1].imshow(decompressed)
plt.show()
