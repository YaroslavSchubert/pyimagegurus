import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True, help='Path to image')
ap.add_argument('-d','--display', required=False)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
if args['display']:
    cv2.imshow("Original", image)
    cv2.waitKey(0)

(b, g, r) = image[0, 0]
print "Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b)

#setting pixel value to RED
image[0,0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print "Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b)


#center of the image
(cX, cY) = (w/2, h/2)
tl = image[0:cY, 0:cX]
tr = image[cY:h, cX:w]
bl = image[cY:h, 0:cX]
tl_green = image[0:cY, 0:cX] = (0, 255, 0)


# cv2.imshow("",tr)
# cv2.waitKey()
# cv2.imshow("",bl)
# cv2.waitKey()
# cv2.imshow("",tl_green)
# cv2.waitKey()
# cv2.imshow("top left", tl)
# cv2.waitKey()