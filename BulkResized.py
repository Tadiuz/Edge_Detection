# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:48:22 2020

@author: j_nba
"""

import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
    
    
protoPath = 'models/deploy.prototxt'
modelPath = 'hed_pretrained_bsds.caffemodel'
dir_path = os.path.join(os.getcwd(),'Images')

net1 = cv.dnn.readNetFromCaffe(protoPath, modelPath)
cv.dnn_registerLayer('Crop', CropLayer)

H = 300

W = 300


for filename in os.listdir(dir_path):

    net = net1
    
    image = cv.imread(os.path.join(dir_path,filename))

    image=cv.resize(image,(W,H))
    
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    # edges = cv.Canny(image,image.shape[1],image.shape[0])
    out = net.forward()
    
    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))
    
    # print(out.shape)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)
    
    
    
    cv.imwrite(f'Output/{filename}', out)