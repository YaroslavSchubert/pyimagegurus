import cv2


class DistanceFinder:
    def __init__(self, known_width, known_distance):
        self.knownWidth = known_width
        self.knownDistance = known_distance
        self.focalLength = 0

    def calibrate(self, width):
        self.focalLength = (width * self.knownDistance) / self.knownWidth

    def distance(self, perceived_width):
        return (self.knownWidth * self.focalLength) / perceived_width


    @staticmethod
    def find_square_marker(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        marker_dim = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.9 < aspect_ratio < 1.1:
                    marker_dim = (x, y, w, h)
                    break

        return marker_dim

    @staticmethod
    def draw(image, bounding_box, dist, color=(0,255,0), thickness=2):
        (x, y, w, h) = bounding_box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(
            image, "%.2fft" % (dist / 12),
            (image.shape[1] - 200, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            color,
            3
        )

        return image

