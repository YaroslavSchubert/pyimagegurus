
# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
from scipy import io
import numpy as np
import argparse
import glob


widths = []
heights = []
conf = Conf('quiz_conf.json')
for p in glob.glob(conf['image_annotations'] + "/*.mat"):
    print(p)
    (y, h, x, w) = io.loadmat(p)['box_coord'][0]
    widths.append(w - x)
    heights.append(h - y)


print(np.mean(widths))
print(np.mean(heights))
