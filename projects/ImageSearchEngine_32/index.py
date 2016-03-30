from __future__ import print_function
from cbir.hsvdescriptor import HSVDescriptor
from imutils import paths
import argparse
import cv2
import tqdm
import multiprocessing
import time

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-i", "--index", required=True, help="where to store features")
ap.add_argument("-m", "--multicpu", required=False, default=False, help='usemultiplecores')

args = vars(ap.parse_args())
desc = HSVDescriptor((4, 6, 3))
output = open(args["index"], "w")
imagePaths = list(paths.list_images(args["dataset"]))


def handler(path):
    return desc.describe(cv2.imread(path))


def index_images_multicore():
    pool = multiprocessing.Pool(6)
    results = pool.map(handler, imagePaths)
    with open(args["index"], "w") as resfile:
        for r, imagePath in zip(results, imagePaths):
            filename = imagePath[imagePath.rfind('/') + 1:]
            features = [str(x) for x in r]
            resfile.write("{},{}\n".format(filename, ",".join(features)))


def index_images_singlecore():
    for im, imagePath in tqdm.tqdm(enumerate(imagePaths)):
        filename = imagePath[imagePath.rfind('/') + 1:]
        image = cv2.imread(imagePath)
        features = desc.describe(image)
        features = [str(x) for x in features]
        output.write("{},{}\n".format(filename, ",".join(features)))


if __name__ == '__main__':
    start = time.clock()
    if args["multicpu"]:
        index_images_multicore()
    else:
        index_images_singlecore()
    print("[INFO] indexed images {0} in {1} [s]".format(len(imagePaths), (time.clock()-start)*10))
