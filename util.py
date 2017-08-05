from PIL import Image, ImageDraw
import math
import glob
import cv2
import numpy

def avg_image(path):
    counter = 0
    mixer = numpy.zeros((288, 384))

    for filename in glob.glob(path):
        img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        ret, bin_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        hl_rate = bin_img.sum() * 1.0 / (img.shape[0] * img.shape[1])
        if hl_rate < 0.12:
            continue
        print 'processing', filename, hl_rate
        counter += 1
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        mixer = mixer + img

    mixer = mixer / counter

    return mixer.astype(numpy.uint8)


if __name__ == '__main__':
    img = avg_image('../DEV00057/*/*.jpg')
    cv2.imwrite('avg.bmp', img)
