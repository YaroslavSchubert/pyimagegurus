import cv2
import numpy as np

class LicensePlateDetector(object):

    def __init__(self, image, minPlateW=60, minPlateH=20):
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH

    def detect(self):
        return self._detectPlates()

    def _detectPlates(self):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []
        coord = []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # !      blackhat is used to reveal dark regions against light backgrounds

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        #  sobel gradient is calculatet to reveal regions that are not only dark
        #  agains light background but also contain veritcal changes in gradient
        gradX = cv2.Sobel(blackhat, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        minVal, maxVal = np.min(gradX), np.max(gradX)
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype('uint8')

        #   gasuain blur removes noise from the image
        #   closing with rect_kernel is used to close regions of the licence plate
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # get rid of irrelevant regions
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # bit wise and to keep only thresholded regions of the image that are
        # also brither than the rest of the image (using the light mask )
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)[:]
            aspectRatio = w / float(h)
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect))
            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
                regions.append(box)
                coord.append((x,y,w,h))

        return regions, coord


