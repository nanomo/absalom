#!/usr/bin/python
# encoding:utf-8 

import sys
import cv2
import numpy
import sys
from glob import glob

class PeopleCount(object):
    def __init__(self):
        self.model = None

    def load_model(self, filename):
        model = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        if model is None:
            raise Exception("load model file %s failed" % filename)
        self.model = model

    def count(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (768, 576), interpolation=cv2.INTER_CUBIC)
        output = cv2.imread(filename)
        output = cv2.resize(output, (768, 576), interpolation=cv2.INTER_CUBIC)
        people = set()
        axis = []
        for y in range(0, 576):
            for x in range(0, 768):
                if self.model[y][x] > 0 and img[y][x] >= 200:
                    if self.model[y][x] not in people:
                        axis.append((x, y))
                        people.add(self.model[y][x])
        for a in axis:
            cv2.circle(output, (a[0], a[1]), 4, (0, 0, 255), 1)
        cv2.imwrite('x.bmp', output)

        return len(people)

if __name__ == '__main__':
    pc = PeopleCount()
    pc.load_model('model.bmp')
    #print 'estimated audience number:', pc.count(sys.argv[1])
    for filename in glob('/Users/duanmiao/Desktop/people_count_project/data/labeled_images/*.jpg'):
        ret = pc.count(filename)
        print filename.replace('/Users/duanmiao/Desktop/people_count_project/data/labeled_images/', ''), ret
