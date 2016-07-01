from baseindexer import BaseIndexer
from scipy import sparse
import numpy as np
import h5py


class BOVWIndexer(BaseIndexer):
    def __init__(self, fvectorSize, dbPath, estNumImages, maxBufferSize=500, dbResizeFactor=2, verbose=2):
        super(BOVWIndexer, self).__init__(dbPath, estNumImages=estNumImages, maxBufferSize=maxBufferSize,
                                          dbResizeFactor=dbResizeFactor, verbose=verbose)
        self.db = h5py.File(self.dbPath)
        self.bovwDB = None
        self.bovwBuffer = None
        self.idxs = {"bovw": 0}

        self.fvectorSize = fvectorSize
        self._df = np.zeros((fvectorSize,), dtype="float")
        self.totalImages = 0

    def add(self, hist):
        # update the BOVW buffer and the document frequency counts
        self.bovwBuffer = BaseIndexer.featureStack(hist, self.bovwBuffer,
                                                   stackMethod=sparse.vstack)
        self._df[np.where(hist.toarray()[0] > 0)] += 1

        # check to see if we have reached the maximum buffer size
        if self.bovwBuffer.shape[0] >= self.maxBufferSize:
            # if the databases have not been created yet, create them
            if self.bovwDB is None:
                self._debug("initial buffer full")
                self._createDatasets()

            # write the buffers to file
            self._writeBuffers()

    def _writeBuffers(self):
        # only write the buffer if there are entries in the buffer
        if self.bovwBuffer is not None and self.bovwBuffer.shape[0] > 0:
            # write the BOVW buffer to file, increment the index, and reset
            # the buffer
            self._writeBuffer(self.bovwDB, "bovw", self.bovwBuffer, "bovw",
                              sparse=True)
            self.idxs["bovw"] += self.bovwBuffer.shape[0]
            self.bovwBuffer = None

    def _createDatasets(self):
        self._debug("creating datasets ... ")
        self.bovwDB = self.db.create_dataset(
            "bovw",
            (self.estNumImages, self.fvectorSize),
            maxshape=(None, self.fvectorSize),
            dtype="float"
        )

    def finish(self):
        # if the databases have not been initialized, then the original
        # buffers were never filled up
        if self.bovwDB is None:
            self._debug("minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()

        # write any unempty buffers to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        # compact datasets
        self._debug("compacting datasets...")

        # TODO FIGURE OUT WHY THIS RESIZING FAILS
        self._resizeDataset(self.bovwDB, "bovw", finished=self.idxs["bovw"])

        # store the total number of images in the dataset and close the
        # database
        self.totalImages = self.bovwDB.shape[0]
        self.db.close()
