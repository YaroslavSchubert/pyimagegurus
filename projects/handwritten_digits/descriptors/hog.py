from skimage import feature


class HOG:
    def __init__(self, orientations=12, pixels_per_cell=(4, 4), cells_per_block=(2, 2), normalize=True):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.normalize = normalize

    def describe(self, image):
        hist = feature.hig(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            transform_sqrt=self.normalize
        )

        hist[hist < 0] = 0
        return hist
