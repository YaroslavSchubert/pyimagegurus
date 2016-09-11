import cv2
import numpy as np
import matplotlib.pyplot as plt

class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, descs) = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        # return a tuple of the keypoints and descriptors
        return (kps, descs)


class CreateFeatureDetector(object):
    def __init__(self, detector_name):
        if cv2.__version__.startswith('3'):
            try:
                cv2.xfeatures2d
            except AttributeError:
                raise ValueError('CV2 ADDON NOT INSTALLED')

            if detector_name == 'FAST':
                self.detector = cv2.FastFeatureDetector_create()

            elif detector_name == 'SURF':
                self.detector = cv2.xfeatures2d.SURF_create()

            elif detector_name == 'SIFT':
                self.detector = cv2.xfeatures2d.SIFT_create()

            else:
                raise ValueError('Unknown detector')

        else:
            raise ValueError('CV version 2.X not supported')

    def detect(self, image):
        return self.detector.detect(image)


class CreateFeatureDescriptor(object):
    def __init__(self, descriptor_name):
        if cv2.__version__.startswith('3'):
            try:
                cv2.xfeatures2d
            except AttributeError:
                raise ValueError('CV2 ADDON NOT INSTALLED')

            if descriptor_name == 'SIFT':
                self.descriptor = cv2.xfeatures2d.SIFT_create()

            elif descriptor_name == 'SURF':
                self.descriptor = cv2.xfeatures2d.SURF_create()

            elif descriptor_name == 'ROOTSIFT':
                self.descriptor = RootSIFT()

            elif descriptor_name == 'BINARY_X':
                raise NotImplementedError

            else:
                raise NotImplementedError

        else:
            raise ValueError('CV version 2.X not supported')

    def compute(self, image):
        self.descriptor.compute(image)

    def describe(self, image, useKpList=True):
        keypoints = self.detect(image)
        if not keypoints:
            return [], None
        _, descriptors = self.descriptor.compute(image, keypoints)

        if len(keypoints) == 0:
            return (None, None)

        # check to see if the keypoints should be converted to a NumPy array
        if useKpList:
            keypoints = np.int0([kp.pt for kp in keypoints])

        # return a tuple of the keypoints and descriptors

        return keypoints, descriptors


class DetectDescribe(CreateFeatureDetector, CreateFeatureDescriptor):
    def __init__(self, detector_name, descriptor_name):
        CreateFeatureDetector.__init__(self, detector_name=detector_name)
        CreateFeatureDescriptor.__init__(self, descriptor_name=descriptor_name)

    def visualize_matches(self, image1, image2, matcher='brute'):
        kps_1, feat_1 = self.describe(image1)
        kps_2, feat_2 = self.describe(image2)

        if matcher == 'brute':
            matcher = cv2.DescriptorMatcher_create('BruteForce')

        raw_matches = matcher.knnMatch(feat_1, feat_2, 2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        (hA, wA) = image1.shape[:2]
        (hB, wB) = image2.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image1
        vis[0:hB, wA:] = image2

        # loop over the matches
        for (trainIdx, queryIdx) in matches:
            # generate a random color and draw the match

            color = np.random.randint(0, high=255, size=(3,))
            ptA = (int(kps_1[queryIdx].pt[0]), int(kps_1[queryIdx].pt[1]))
            ptB = (int(kps_2[trainIdx].pt[0] + wA), int(kps_2[trainIdx].pt[1]))
            cv2.line(vis, ptA, ptB, color, 1)

        plt.figure()
        return plt.imshow(vis)








