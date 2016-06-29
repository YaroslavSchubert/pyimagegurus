# import the necessary packages
from __future__ import print_function
from ir.bovw import BagOfVisualWords
from sklearn.metrics import pairwise
import numpy as np

# randomly generate the vocabulary/cluster centers along with the feature
# vectors -- we'll generate 10 feature vectors containing 6 real-valued
# entries, along with a codebook containing 3 'visual words'
np.random.seed(42)


vocab = np.random.uniform(size=(3, 6))
features = np.random.uniform(size=(10, 6))

bvw = BagOfVisualWords(codebook=vocab, sparse=False)
hist = bvw.describe(features)
print(hist)