import numpy as np
import cv2


class PCDLib:

    # =========================
    # FILTER SPASIAL
    # =========================

    @staticmethod
    def mean_filter(img, size=3):

        pad = size // 2
        padded = np.pad(img, pad, mode='edge')

        h, w = img.shape
        hasil = np.zeros_like(img)

        area = size * size

        for i in range(h):
            for j in range(w):

                region = padded[i:i+size, j:j+size]

                hasil[i, j] = np.sum(region) / area

        return hasil.astype(np.uint8)

    @staticmethod
    def median_filter(img, size=3):

        pad = size // 2
        padded = np.pad(img, pad, mode='edge')

        h, w = img.shape
        hasil = np.zeros_like(img)

        for i in range(h):
            for j in range(w):

                region = padded[i:i+size, j:j+size]

                hasil[i, j] = np.median(region)

        return hasil.astype(np.uint8)

    @staticmethod
    def mode_filter(img, size=3):

        pad = size // 2
        padded = np.pad(img, pad, mode='edge')

        h, w = img.shape
        hasil = np.zeros_like(img)

        for i in range(h):
            for j in range(w):

                region = padded[i:i+size, j:j+size]

                values = region.ravel()

                count = {}

                for val in values:

                    if val in count:
                        count[val] += 1
                    else:
                        count[val] = 1

                mode_val = max(count, key=count.get)

                hasil[i, j] = mode_val

        return hasil.astype(np.uint8)

    # =========================
    # KONVOLUSI
    # =========================

    @staticmethod
    def convolution(img, kernel):

        size = kernel.shape[0]
        pad = size // 2

        padded = np.pad(img, pad, mode='constant')

        h, w = img.shape
        hasil = np.zeros_like(img).astype(np.float32)

        for i in range(h):
            for j in range(w):

                region = padded[i:i+size, j:j+size]

                hasil[i, j] = np.sum(region * kernel)

        return hasil

    # =========================
    # SMOOTHING & SHARPENING
    # =========================

    @staticmethod
    def smoothing(img):

        kernel = np.array([
            [1/10, 1/10, 1/10],
            [1/10, 1/5,  1/10],
            [1/10, 1/10, 1/10]
        ])

        hasil = PCDLib.convolution(img, kernel)

        return np.clip(hasil, 0, 255).astype(np.uint8)

    @staticmethod
    def sharpening(img):

        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        hasil = PCDLib.convolution(img, kernel)

        return np.clip(hasil, 0, 255).astype(np.uint8)

    # =========================
    # NORMALISASI
    # =========================

    @staticmethod
    def normalize(img):

        img = np.abs(img)

        img = (img / img.max()) * 255

        return img.astype(np.uint8)

    # =========================
    # SOBEL
    # =========================

    @staticmethod
    def sobel(img):

        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        gx = PCDLib.convolution(img, sobel_x)
        gy = PCDLib.convolution(img, sobel_y)

        hasil = np.abs(gx) + np.abs(gy)

        return PCDLib.normalize(hasil)

    # =========================
    # PREWITT
    # =========================

    @staticmethod
    def prewitt(img):

        prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        prewitt_y = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ])

        gx = PCDLib.convolution(img, prewitt_x)
        gy = PCDLib.convolution(img, prewitt_y)

        hasil = np.abs(gx) + np.abs(gy)

        return PCDLib.normalize(hasil)

    # =========================
    # ROBERTS
    # =========================

    @staticmethod
    def roberts(img):

        roberts_x = np.array([
            [1, 0],
            [0,-1]
        ])

        roberts_y = np.array([
            [0, 1],
            [-1,0]
        ])

        gx = PCDLib.convolution(img, roberts_x)
        gy = PCDLib.convolution(img, roberts_y)

        hasil = np.abs(gx) + np.abs(gy)

        return PCDLib.normalize(hasil)

    # =========================
    # UTILITAS
    # =========================

    @staticmethod
    def grayscale(img):

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def read(path, gray=False):

        if gray:
            return cv2.imread(path, 0)

        return cv2.imread(path)

    @staticmethod
    def save(path, img):

        cv2.imwrite(path, img)