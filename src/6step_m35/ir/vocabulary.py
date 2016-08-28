from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py


class Vocabulary(object):
    def __init__(self, dbPath, verbose=True):
        self.dbPath = dbPath
        self.verbose = verbose

    def fit(self, numClusters, samplePercent, randomState=None):
        db = h5py.File(self.dbPath)
        totalFeatures = db['features'].shape[0]
        sampleSize = int(np.ceil(samplePercent * totalFeatures))

        idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling ... ")

        for i in idxs:
            data.append(db['features'][i][2:])

        # cluster the data
        self._debug("sampled {:,} features from a population of {:,}".format(
            len(idxs), totalFeatures))
        self._debug("clustering with k={:,}".format(numClusters))
        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        clt.fit(data)
        self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

        db.close()

        return clt.cluster_centers_

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))



