#!/usr/bin/python
# encoding:utf-8 

import sys
import cv2
import numpy
import sys

class PeopleCount(object):
    def __init__(self):
        self.model = None

    def load_model(self, filename):
        model = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        if model is None:
            raise Exception("load model file %s failed" % filename)
        self.model = model

    def count(self, filename):
        img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (768, 576), interpolation=cv2.INTER_CUBIC)
        output = numpy.zeros((576, 768, 3), numpy.uint8)
        people = set()
        axis = []
        for y in range(0, 576):
            for x in range(0, 768):
                output[y][x] = (img[y][x], img[y][x], img[y][x])
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
    print 'estimated audience number:', pc.count(sys.argv[1])
