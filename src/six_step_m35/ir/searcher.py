from searchresults import SearchResult
import numpy as np
import datetime
import dists
import h5py


class Searcher(object):
    def __init__(self, redisDB, bovwDBPath, featuresDBPath, idf=None, distanceMetric=dists.chi2_distance):
        self.redisDB = redisDB
        self.idf = idf
        self.distanceMetric = distanceMetric

        self.bovwDB = h5py.File(bovwDBPath, mode='r')
        self.featuresDB = h5py.File(featuresDBPath, 'r')

    def search(self, queryHist, numResults=10, maxCandidates=200):
        startTime = datetime.datetime.now()
        candidateIdxs = self.buildCandidates(queryHist, maxCandidates)
        candidateIdxs.sort()

        hists = self.bovwDB['bovw'][candidateIdxs]
        queryHist = queryHist.toarray()
        results = {}

        if self.idf is not None:
            queryHist *= self.idf

        for (candidate, hist) in zip(candidateIdxs, hists):
            if self.idf is not None:
                hist *= self.idf

            d = self.distanceMetric(hist, queryHist)
            results[candidate] = d

        # sort the results this time replacing image
        # indexes with the image
        # IDs themselves

        results = sorted(
            [
                (v, self.featuresDB['image_ids'][k], k)
                for (k, v) in results.items()]
        )

        results = results[:numResults]
        return SearchResult(results, (datetime.datetime.now() - startTime).total_seconds())

    def buildCandidates(self, hist, maxCandidates):
        p = self.redisDB.pipeline()
        for i in hist.col:
            p.lrange("vw:{}".format(i), 0, -1)

        pipelineResults = p.execute()
        candidates = []

        for results in pipelineResults:
            result = [int(r) for r in results]
            candidates.extend(result)

        (imageIdxs, counts) = np.unique(candidates, return_counts=True)
        imageIdxs = [i for (c, i) in sorted(zip(counts, imageIdxs), reverse=True)]

        return imageIdxs[:maxCandidates]

    def finish(self):
        # close the bag-of-visual-words database and the features database
        self.bovwDB.close()
        self.featuresDB.close()
# TODO: snap!
