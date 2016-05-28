from skimage import feature
import numpy as np


class LocalBinaryPatterns(object):
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + eps)
        return hist





from concurrent.futures import ProcessPoolExecutor


L = [1,2,3,4,5]

with ProcessPoolExecutor(6) as executor:
    results = executor.map(lambda x: x**3, L)