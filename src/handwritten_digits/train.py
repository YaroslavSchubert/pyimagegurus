# import the necessary packages
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from descriptors.hog import HOG
from utils import dataset
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset file")
ap.add_argument("-m", "--model", required=True, help="path to where the model will be stored")
args = vars(ap.parse_args())