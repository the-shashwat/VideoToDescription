# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

im1 = np.array(Image.open('examples/S_F0.jpg'))
im2 = np.array(Image.open('examples/S_F1.jpg'))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.
print('im1 shape ',im1.shape)
print('im2 shape ',im2.shape)
# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
e = time.time()
print('u dimension ',u.shape)
print('v dimension ',v.shape)
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save('examples/outFlow.npy', flow)
print('flow shape ',flow.shape)
#print('u----------->')
#print(u)
#print("v----------->")
#print(v)
print("flow-------->")
print(flow)
if args.viz:
    import cv2

#    hsv = np.zeros(im1.shape, dtype=np.uint8)
#    hsv[:, :, 0] = 255
#    hsv[:, :, 1] = 255
#    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#    print("Magnitude dimension ",mag.shape)
#    print("Ang dimension ",ang.shape)
#    print("Mag ----->")
#    print(mag)
#    print('Ang ----->')
#    print(ang)
#    hsv[..., 0] = ang * 180 / np.pi / 2
#    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    max_flow = 8
    scale = 128/max_flow

    mag_flow = np.sqrt(np.square(flow[..., 0])+np.square(flow[..., 1]))
    flow = flow*scale  
    flow = flow+128 
    flow[flow<0] = 0
    flow[flow>255] = 255

    mag_flow = mag_flow*scale
    mag_flow = mag_flow+128
    mag_flow[mag_flow<0] = 0
    mag_flow[mag_flow>255] = 255
 
    flow_img = np.dstack([flow,mag_flow[...,np.newaxis]])
    flow_img = flow_img.astype(int)
    cv2.imwrite('examples/S_F191_192_flow_star.png', flow_img)
    #cv2.imwrite('examples/S_F_warped.jpg', im2W[:, :, ::-1] * 255)
