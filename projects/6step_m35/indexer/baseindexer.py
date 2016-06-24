from __future__ import print_function
import numpy as np
import datetime


class BaseIndexer(object):
    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2, verbose=True):
        self.dbPath = dbPath
        self.estNumImages = estNumImages
        self.maxBufferSize = maxBufferSize
        self.dbResizeFactor = dbResizeFactor
        self.verbose = verbose
        self.idxs = {}

    def _writeBuffers(self):
        # write the buffers to disk
        self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer,
                          "index")
        self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
        self._writeBuffer(self.featuresDB, "features", self.featuresBuffer,
                          "features")

        # increment the indexes
        self.idxs["index"] += len(self.imageIDBuffer)
        self.idxs["features"] += self.totalFeatures

        # reset the buffers and feature counts
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.featuresBuffer = None
        self.totalFeatures = 0

    def _writeBuffer(self, dataset, datasetName, buf, idxName, sparse=False):
        # if the buffer is a list, then compute the ending index based on
        # the lists length
        if type(buf) is list:
            end = self.idxs[idxName] + len(buf)

        # otherwise, assume that the buffer is a NumPy/SciPy array, so
        # compute the ending index based on the array shape
        else:
            end = self.idxs[idxName] + buf.shape[0]

        # check to see if the dataset needs to be resized
        if end > dataset.shape[0]:
            self._debug("triggering `{}` db resize".format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)

        # if this is a sparse matrix, then convert the sparse matrix to a
        # dense one so it can be written to file
        if sparse:
            buf = buf.toarray()

        # if datasetName == 'features':
        #     import pdb
        #     pdb.set_trace()
        # dump the buffer to file
        self._debug("writing `{}` buffer".format(datasetName))
        dataset[self.idxs[idxName]:end] = buf

    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
        origSize = dataset.shape[0]

        if finished > 0:
            newSize = finished
        else:
            newSize = baseSize * self.dbResizeFactor

        shape = list(dataset.shape)
        shape[0] = newSize

        dataset.resize(tuple(shape))
        self._debug("old size of {}: {:,}; new size: {:,}".format(dbName, origSize, newSize))

    def _debug(self, msg, msgType="[INFO]"):
        # check to see if the message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    @staticmethod
    def featureStack(array, accum=None, stackMethod=np.vstack):
        if accum is None:
            accum = array

        else:
            accum = stackMethod([accum, array])

        return accum
