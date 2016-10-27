from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import numpy as np


class BagOfVisualWords(object):
    def __init__(self, codebook, sparse=True):
        self.codebook = codebook
        self.sparse = sparse

    def describe(self, features):
        """
        # compute the Euclidean distance between the features and cluster centers,
        # grab the indexes of the smallest distances for each cluster, and construct
        # a bag-of-visual-words representation

        :param features:
        :return:
        """

        D = pairwise.euclidean_distances(features, Y=self.codebook)
        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)

        if self.sparse:
            hist = csr_matrix(
                (counts, (np.zeros((len(words),)), words)), shape=(1, len(self.codebook)), dtype="float"
            )

        else:
            hist = np.zeros((len(self.codebook),), dtype="float")
            hist[words] = counts

        return hist


if __name__ == '__main__':
    # np.random.seed(84)
    # vocab = np.random.uniform(size=(3, 36))
    # # vocab = np.random.uniform(size=(3, 6))
    # features = np.random.uniform(size=(10, 6))
    #
    # bvw = BagOfVisualWords(codebook=vocab, sparse=False)
    # hist = bvw.describe(features)
    #
    # np.random.seed(84)
    # vocab = np.random.uniform(size=(3, 36))
    # features = np.random.uniform(size=(100, 36))
    # bovw = BagOfVisualWords(vocab, sparse=False)
    # hist = bovw.describe(features)
    # print("[INFO] BOVW histogram: {}".format(hist))

    np.random.seed(42)
    vocab = np.random.uniform(size=(5, 36))
    features = np.random.uniform(size=(500, 36))
    bovw = BagOfVisualWords(vocab, sparse=False)
    hist = bovw.describe(features)
    print("[INFO] BOVW histogram: {}".format(hist))