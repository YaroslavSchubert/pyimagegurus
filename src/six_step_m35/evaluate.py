from descriptors.detectanddescribe import DetectDescribe
from ir.bovw import BagOfVisualWords
from ir.searcher import SearchResult, Searcher
from ir import dists
from scipy.spatial import distance
from redis import Redis
import numpy as np
import progressbar
import argparse
import cPickle
import imutils
import json
import cv2


#python evaluate.py --dataset ../../data/ukbench --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 --codebook output/vocab.cpickle --idf output/idf.cpickle --relevant ../../data/ukbench/relevant.json
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help="Path to relevant dictionary")
args = vars(ap.parse_args())

dad = DetectDescribe("SURF", "ROOTSIFT")
distanceMetric = dists.chi2_distance
idf = None

if args['idf'] is not None:
    idf = cPickle.loads(open(args['idf']).read())
    distanceMetric = distance.cosine

vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)
redisDB = Redis(host='localhost', port=6379, db=11)
searcher = Searcher(redisDB, args['bovw_db'], args['features_db'], idf=idf, distanceMetric=distanceMetric)
relevant = json.loads(open(args['relevant']).read())
queryIDs = relevant.keys()

accuracies = []
timings = []
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queryIDs), widgets=widgets).start()

# loop over the images
for (i, queryID) in enumerate(sorted(queryIDs)):
    # look up the relevant results for the query image
    queryRelevant = relevant[queryID]

    # load the query image and process it
    p = "{}/{}".format(args["dataset"], queryID)
    queryImage = cv2.imread(p)
    queryImage = imutils.resize(queryImage, width=320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct a bag-of-visual-words
    # from it
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    # perform the search and compute the total number of relevant images in the
    # top-4 results
    sr = searcher.search(hist, numResults=4)
    results = set([r[1] for r in sr.results])
    inter = results.intersection(queryRelevant)

    # update the evaluation lists
    accuracies.append(len(inter))
    timings.append(sr.search_time)
    pbar.update(i)

# release any pointers allocated by the searcher
searcher.finish()
pbar.finish()

# show evaluation information to the user
accuracies = np.array(accuracies)
timings = np.array(timings)
print("[INFO] ACCURACY: u={:.2f}, o={:.2f}".format(accuracies.mean(), accuracies.std()))
print("[INFO] TIMINGS: u={:.2f}, o={:.2f}".format(timings.mean(), timings.std()))
