from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import zlib

BLOCK_SIZE = 8
Q_MATRIX = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


# RGB -> YCrCb
def to_YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


# YCrCb -> RGB
def from_YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)


# Applies dct and then coefficient quantization over a 8*8 block.
# Returns a 8*8 block of np.int32.
def quantize_block(block, debug=False):
    assert type(block) == np.ndarray
    assert block.shape == (8, 8)
    assert block.dtype == np.uint8

    # Shift the values from [0, 255] to [-128, 127]
    shifted_block = block.astype(np.int32)
    shifted_block -= 128

    # Take the two-dimensional DCT
    y = dctn(shifted_block, orthogonalize=True) / 8

    # Coefficient quantization
    y_q = (y / Q_MATRIX).astype(np.int8)

    if debug:
        block_q = dequantize_block(y_q)
        # To see the compression result side-by-side:
        fig, axes = plt.subplots(2, 2)
        fig.tight_layout()
        axes[0, 0].set_title("The block")
        axes[0, 0].imshow(block)
        axes[0, 1].set_title("The block after DCT quantization")
        axes[0, 1].imshow(block_q)

        axes[1, 0].set_title("The DCT")
        axes[1, 0].imshow(y)
        axes[1, 1].set_title("The DCT after quantization")
        axes[1, 1].imshow(y_q)
        plt.show()

    return y_q


def dequantize_block(block):
    assert type(block) == np.ndarray
    assert block.shape == (8, 8)
    assert block.dtype == np.int8

    block = block.astype(np.int32) * Q_MATRIX * 8
    block = np.round(idctn(block, orthogonalize=True))
    block += 128
    block = np.clip(block, 0, 255)
    block = block.astype(np.uint8)
    return block


# Compress a 8*8 block into a byte array.
def compress_block(block):
    assert block.shape == (8, 8)
    assert block.dtype == np.int8

    data = block.reshape(64).tobytes()
    return zlib.compress(data)


# Decompresses a byte array into a 8*8 block.
def decompress_block(data):
    data = zlib.decompress(data)
    block = np.frombuffer(data, dtype=np.int8).reshape(8, 8)
    return block


class CompressedGrayscaleImage:
    def __init__(self, height, width, padded_height, padded_width, blocks):
        self.height = height
        self.width = width
        self.padded_height = padded_height
        self.padded_width = padded_width
        self.blocks = blocks

    def size(self):
        return sum(map(lambda k: len(k), self.blocks))


class CompressedRGBImage:
    def __init__(self, compressed_chan1, compressed_chan2, compressed_chan3):
        self.compressed_chan1 = compressed_chan1
        self.compressed_chan2 = compressed_chan2
        self.compressed_chan3 = compressed_chan3

    def size(self):
        return (
            self.compressed_chan1.size()
            + self.compressed_chan2.size()
            + self.compressed_chan3.size()
        )


# Compresses a grayscale image using the JPEG algorithm.
def compress_grayscale(image):
    assert type(image) == np.ndarray
    assert len(image.shape) == 2

    height, width = image.shape
    # Pad the image so that we can split it into
    # blocks of 8x8
    pad_height, pad_width = height % 8, width % 8
    if pad_height != 0:
        image = np.pad(image, ((0, 8 - pad_height), (0, 0)), mode="edge")
    if pad_width != 0:
        image = np.pad(image, ((0, 0), (0, 8 - pad_width)), mode="edge")

    new_height, new_width = image.shape
    blocks = []

    for y in range(0, new_height, BLOCK_SIZE):
        for x in range(0, new_width, BLOCK_SIZE):
            block = image[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            quantized_block = quantize_block(block, debug=False)

            block_data = compress_block(quantized_block)
            blocks.append(block_data)

    return CompressedGrayscaleImage(height, width, new_height, new_width, blocks)


def decompress_grayscale(image):
    assert type(image) == CompressedGrayscaleImage
    decompressed = np.zeros((image.padded_height, image.padded_width), dtype=np.uint8)
    current_block = 0

    for y in range(0, image.padded_height, BLOCK_SIZE):
        for x in range(0, image.padded_width, BLOCK_SIZE):
            block_data = image.blocks[current_block]
            quantized_block = decompress_block(block_data)
            block = dequantize_block(quantized_block)
            decompressed[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = block
            current_block += 1
    return decompressed[: image.height, : image.width]


def compress_rgb(image):
    assert type(image) == np.ndarray
    assert len(image.shape) == 3
    assert image.shape[2] == 3

    image = to_YCrCb(image)
    c1, c2, c3 = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    compressed_chan1 = compress_grayscale(c1)
    compressed_chan2 = compress_grayscale(c2)
    compressed_chan3 = compress_grayscale(c3)

    return CompressedRGBImage(compressed_chan1, compressed_chan2, compressed_chan3)


def decompress_rgb(image):
    assert type(image) == CompressedRGBImage

    c1 = decompress_grayscale(image.compressed_chan1)
    c2 = decompress_grayscale(image.compressed_chan2)
    c3 = decompress_grayscale(image.compressed_chan3)

    image = np.stack((c1, c2, c3), axis=-1)

    return from_YCrCb(image)
