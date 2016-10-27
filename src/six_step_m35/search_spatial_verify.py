# USAGE
# WITHOUT TF-IDF;
# python search.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 \
#	--codebook output/vocab.cpickle --relevant ~/Desktop/ukbench_sample/relevant.json \
#	--query ~/Desktop/ukbench_sample/ukbench00760.jpg
#
# WITH TF-IDF:
# python search.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 \
# 	--codebook output/vocab.cpickle --idf output/idf.cpickle --relevant ../data/ukbench/relevant.json
# 	--query ../data/ukbench/ukbench00760.jpg

# import the necessary packages
from __future__ import print_function
from src.descriptors.detectanddescribe import DetectDescribe
from src.descriptors.rootsift import RootSIFT
from src.six_step_m35.ir.searcher import SearchResult, Searcher
# from descriptors import RootSIFT
from ir.bovw import BagOfVisualWords
from ir.searcher import Searcher
from ir import dists
from src.ImageSearchEngine_32.cbir.resultsmontage import ResultsMontage
from scipy.spatial import distance
from redis import Redis
import argparse
import cPickle
import imutils
import json
import cv2

# construct the argument parser and parse the arguments
from src.six_step_m35.ir.spatialverifier import SpatialVerifier

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help="Path to relevant dictionary")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())

# # initialize the keypoint detector, local invariant descriptor, and descriptor pipeline
# detector = cv2.FeatureDetector_create("SURF")
# descriptor = RootSIFT()
# dad = DetectDescribe(detector, descriptor)
dad = DetectDescribe("SURF", "ROOTSIFT")

# load the inverted document frequency array and codebook vocabulary, then
# initialize the bag-of-visual-words transformer
idf = cPickle.loads(open(args["idf"]).read())
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)

# load the relevant queries dictionary and look up the relevant results for the
# query image
relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
queryRelevant = relevant[queryFilename]

# load the query image and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width=320))
queryImage = imutils.resize(queryImage, width=320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# extract features from the query image and construct a bag-of-visual-words from it
(queryKps, queryDescs) = dad.describe(queryImage)
queryHist = bovw.describe(queryDescs).tocoo()

# connect to redis and perform the search
redisDB = Redis(host="localhost", port=6379, db=11)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf,
                    distanceMetric=distance.cosine)
sr = searcher.search(queryHist, numResults=20)
print("[INFO] search took: {:.2f}s".format(sr.search_time))

# spatially verify the results
spatialVerifier = SpatialVerifier(args["features_db"], idf, vocab)
sv, m = spatialVerifier.rerank(queryKps, queryDescs, sr, numResults=20)
print("[INFO] spatial verification took: {:.2f}s".format(sv.search_time))

montage = ResultsMontage((240, 320), 5, 20)

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(sv.results):
    # load the result image and display it
    print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num=i + 1,
                                                                 result=resultID, score=score))
    result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    montage.addResult(result, text="#{}".format(i + 1),
                      highlight=resultID in queryRelevant)

# show the output image of results
cv2.imshow("Results", imutils.resize(montage.montage, height=700))
cv2.waitKey(0)
searcher.finish()
spatialVerifier.finish()