import dists
import csv


class Searcher(object):
    def __init__(self, db_path):
        self.db_path = db_path

    def search(self, query_features, numResults=10):
        results = {}
        with open(self.db_path) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = dists.chi2_distance(features, query_features)
                results[row[0]] = d

            f.close()

        results = sorted([(v, k) for k, v in results.items()])
        return results[:numResults]
