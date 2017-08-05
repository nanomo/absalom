# -*- coding=utf-8 -*-
import copy

import numpy as np
import cv2
import math

import util

# max: 40x40
# min: 14x14
# 首先叠加足够多的历史图片，然后识别出mask，对mask使用插值法来补全

MIN_WIDTH = 10
MIN_HEIGHT = 10

def findFeature(img, mask, x, y, maxHeight, counter):
    footsteps = [(y, x)]
    # move to left, or down, or left down
    left = x
    downl = y
    noLeft = False
    while True:
        if downl - y + 1 == maxHeight:
            break
        found = False
        # go left
        if noLeft == False and left >= 1 and img[downl][left-1] > 0 and mask[downl][left-1] == 0:
            # if the curve goes up, just quit!
            if downl > 0 and img[downl-1][left-1] > 0:
                noLeft = True
                continue
            left -= 1
            footsteps.append((downl, left))
            continue
        # go left down
        if noLeft == False and left >= 1 and downl < img.shape[0] - 1 and img[downl+1][left-1] > 0 and mask[downl+1][left-1] == 0:
            left -= 1
            footsteps.append((downl, left))
            downl += 1
            footsteps.append((downl, left))
            continue
        # go down
        if downl < img.shape[0] - 1 and img[downl+1][left] > 0:
            downl += 1
            footsteps.append((downl, left))
            continue
        break

    right = x
    downr = y
    noRight = False
    while True:
        if downr - y + 1 == maxHeight:
            break
        # go right
        if noRight == False and right < img.shape[1]-1 and img[downr][right+1] > 0 and mask[downr][right+1]==0:
            # if the curve goes up, just quit!
            if downr > 0 and img[downr-1][right+1] > 0:
                noRight = True
                continue
            right += 1
            footsteps.append((downr, right))
            continue
        # go right down
        if noRight == False and right < img.shape[1]-1 and downr < img.shape[0]-1 and img[downr+1][right+1] > 0 and mask[downr+1][right+1] == 0:
            right += 1
            footsteps.append((downr, right))
            downr += 1
            footsteps.append((downr, right))
            continue
        # go down
        if downr < img.shape[0] - 1 and img[downr+1][right] > 0:
            downr += 1
            footsteps.append((downr, right))
            continue
        break

    width = right - left + 1

    # 过滤掉不符合要求的轮廓
    if width < MIN_WIDTH:
        return
    height = min(downl, downr) - y
    if height < MIN_HEIGHT:
        return
    if left == x or right == x:
        return

    # colorize by max
    canvas = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for step in footsteps:
        canvas[step[0]][step[1]] = counter
    if downl > downr:
        cv2.line(canvas, (right, downr), (right, downl), counter, 1)
        cv2.line(canvas, (left, downl), (right, downl), counter, 1)
    else:
        cv2.line(canvas, (left, downl), (left, downr), counter, 1)
        cv2.line(canvas, (left, downr), (right, downr), counter, 1)
    fillMask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    centerX = int((left*1.0+right)/2)
    centerY = int((max(downl, downr)*1.0 + y)/2)
#    cv2.circle(canvas, (centerX, centerY), 2, 100, 1)
#    cv2.floodFill(mask, fillMask, (centerX, centerY), 255)
#    cv2.floodFill(canvas, fillMask, (centerX, centerY), 255)
    for maxy in range(y + 1, max(downr, downl)):
        if canvas[maxy][x] == 0:
#            cv2.circle(canvas, (x, maxy), 2, 255, 1)
            cv2.floodFill(canvas, fillMask, (x, maxy), counter)
#            cv2.circle(canvas, (x, maxy), 2, 255, 1)
            break
    return canvas

def markup(avg):

    res = cv2.resize(avg, (768, 576), interpolation=cv2.INTER_LINEAR)

    maxHeightMap = {
        192: 22,
        160: 22,
        128: 22,
        64: 18,
        32: 14,
        20: 14
    }

    mask = np.zeros((res.shape[0], res.shape[1]), np.uint8)

    #for thres in [192, 160, 128, 64, 32, 20]:
    for thres in [192, 160, 128, 64, 32, 20]:
        img = copy.deepcopy(res)
        #img = cv2.blur(img, (3,3))
        ret, img = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)

        # flood fill four corners
        fillMask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        cv2.floodFill(img, fillMask, (0, 0), 0)
        cv2.floodFill(img, fillMask, (img.shape[1]-1, img.shape[0]-1), 0)
        cv2.floodFill(img, fillMask, (0, img.shape[0]-1), 0)
        cv2.floodFill(img, fillMask, (img.shape[1]-1, 0), 0)

        for y in xrange(0, img.shape[0]):
            for x in xrange(0, img.shape[1]):
                # already used 
                if mask[y][x] > 0:
                    continue
                if img[y][x] == 255 and (y == 0 or img[y-1][x] == 0):
                    canvas = findFeature(img, mask, x, y, maxHeightMap.get(thres, 18), 255)
                    if canvas is not None:
                        mask = mask | canvas

        # display result
        if True == True:
            output = cv2.equalizeHist(avg)
            output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for i in xrange(0, output.shape[0]):
                for j in xrange(0, output.shape[1]):
                    if mask[i][j] > 0: #for contour
                        output[i][j] = (0, 0, 255)
            cv2.namedWindow("Image")
            cv2.imshow("Image", output)
            cv2.waitKey(0)

    return mask


def gen_model(img):
    # find and mark
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    counter = 1
    for y in xrange(0, img.shape[0]):
        for x in xrange(0, img.shape[1]):
            if mask[y][x] > 0:
                continue
            if img[y][x] == 255 and (y == 0 or img[y-1][x] == 0):
                canvas = findFeature(img, mask, x, y, 40, counter)
                if canvas is not None:
                    counter += 1
                    mask = mask + canvas
    print '%d seats found' % counter

    # display result
    img = cv2.imread('avg.bmp')
    img = cv2.resize(img, (768, 576), interpolation=cv2.INTER_CUBIC)
#    img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in xrange(0, img.shape[0]):
        for j in xrange(0, img.shape[1]):
            if mask[i][j] > 0: #for contour
                img[i][j] = (0, 0, 255)
    cv2.namedWindow("Image")
    cv2.imshow("Image", mask)
    cv2.waitKey(0)

    cv2.imwrite('model.bmp', mask)

    return mask

if __name__ == '__main__':
#    avg = util.avg_image('../DEV00057/*/*.jpg')
#    cv2.imwrite('avg.bmp', avg)

    avg = cv2.imread('avg.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#    avg = cv2.imread('DEV00057_IR_2017-07-07_18-42-00-339.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

    marked = markup(avg)

#    img = cv2.resize(img, (768, 576), interpolation=cv2.INTER_CUBIC)
##    img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    img = cv2.equalizeHist(img)
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    for i in xrange(0, img.shape[0]):
#        for j in xrange(0, img.shape[1]):
#            if marked[i][j] > 0: #for contour
#                img[i][j] = (0, 0, 255)
#    cv2.namedWindow("Image")
#    cv2.imshow("Image", img)
#    cv2.waitKey(0)

    market = cv2.imread('avg.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gen_model(marked)
