import numpy as np


class DetectAndDescribe(object):
    def __init__(self, detector, descriptor):
        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList=True):
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)

        if len(kps) == 0:
            return None, None

        if useKpList:
            kps = np.int0([kp.pt for kp in kps])

        return kps, descs




class CreateFeatureDetector(object):
    def __init__(self, detector_name):
        if detector_name == 'FAST':
            self.detector = cv2.FastFeatureDetector_create()
        elif detector_name == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()

    def detect(self, image):
        return self.detector.detect(image)


class CreateFeatureDescriptor(object):
    def __init__(self, descriptor_name):
        self.descriptor = descriptor_name
        pass

    def describe(self, image):
        pass

class DetectDescribe(CreateFeatureDetector, CreateFeatureDescriptor):
    def __init__(self, detector_name):
        super(DetectDescribe, self).__init__(detector_name)
