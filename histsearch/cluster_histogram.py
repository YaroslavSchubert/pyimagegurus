from histsearch.descriptors.labhistograms import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to the input dataset directory")
ap.add_argument("-k", "--clusters", type=int, default=2,
    help="# of clusters to generate")
args = vars(ap.parse_args())