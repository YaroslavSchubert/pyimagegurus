# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2.ocl.setUseOpenCL(False)


class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        if cv2.__version__.startswith('3'):
            self.extractor = cv2.xfeatures2d.SIFT_create()
        else:
            self.extractor = cv2.DescriptorExtractor_create("SIFT")

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

            elif detector_name == 'ORB':
                self.detector = cv2.ORB_create()

            else:
                raise ValueError('Unknown detector')

        else:
            if detector_name == "SURF":
                self.detector = cv2.FeatureDetector_create("SURF")
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

            elif descriptor_name == 'BINARY_BRIEF':
                self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

            elif descriptor_name == 'BINARY_ORB':
                self.descriptor = cv2.ORB_create()

            elif descriptor_name == 'BINARY_FREAK':
                self.descriptor = cv2.xfeatures2d.FREAK_create()

            else:
                raise NotImplementedError

        else:
            if descriptor_name == "ROOTSIFT":
                self.descriptor = RootSIFT()

            else:
                raise ValueError('no support for this descriptor in CV2')

        self.descriptor_name = descriptor_name

    def compute(self, image):
        return self.descriptor.compute(image)

    def describe(self, image):
        keypoints = self.detect(image)
        if not keypoints:
            return [], None
        _, descriptors = self.descriptor.compute(image, keypoints)
        if self.use_kp_list:
                keypoints = np.int0([kp.pt for kp in keypoints])
        return keypoints, descriptors


class DetectDescribe(CreateFeatureDetector, CreateFeatureDescriptor):
    def __init__(self, detector_name, descriptor_name, use_kp_list=True):
        self.use_kp_list = True
        CreateFeatureDetector.__init__(self, detector_name=detector_name)
        CreateFeatureDescriptor.__init__(self, descriptor_name=descriptor_name)

    def visualize_matches(self, image1, image2, matcher='brute', distance_threshold=0.8):
        kps_1, feat_1 = self.describe(image1)
        kps_2, feat_2 = self.describe(image2)

        if matcher == 'brute':
            if self.descriptor_name.startswith('BINARY'):
                matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
            else:
                matcher = cv2.DescriptorMatcher_create('BruteForce')

        raw_matches = matcher.knnMatch(feat_1, feat_2, 2)
        matches = []

        for m in raw_matches:
            #         The reason we want the top two matches rather than just the top one match is
            #         because we need to apply David Lowe’s ratio test for false-positive match pruning.
            #         Again, Line 51 computes the rawMatches  for each of the descriptor pairs —
            #         but there is a chance that some of these pairs are false positives, meaning that the patches are not actually matches. In an attempt to prune these false-positive matches, we can loop over each of the rawMatches  individually (Line 56) and apply Lowe’s ratio test, which is used to determine high-quality feature matches.
            #         This test rejects poor matches by computing the ratio between the best and second-best match. If the ratio is above some threshold, the match is discarded as being low quality (Lines 57 and 58). Lowe’s ratio test works well in practice since correct matches need to have the closest neighbor significantly closer than the closest incorrect match to achieve reliable matching. Common values for the ratio r are typically in the range [0.7, 0.8].
            if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        print(len(matches))

        (hA, wA) = image1.shape[:2]
        (hB, wB) = image2.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image1
        vis[0:hB, wA:] = image2

        # loop over the matches
        for (trainIdx, queryIdx) in matches[:1000]:
            # generate a random color and draw the match
            color = np.random.randint(0, high=255, size=(3,))
            ptA = (int(kps_1[queryIdx].pt[0]), int(kps_1[queryIdx].pt[1]))
            ptB = (int(kps_2[trainIdx].pt[0] + wA), int(kps_2[trainIdx].pt[1]))
            cv2.line(vis, ptA, ptB, color, 2)

        plt.figure(figsize=(14, 14))
        return plt.imshow(vis)
