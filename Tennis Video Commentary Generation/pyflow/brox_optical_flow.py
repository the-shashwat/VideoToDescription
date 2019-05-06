# Incorporated the flow algorithm implementation by Deepak Pathak (c) 2016
# Incorporating his implementation into my work on Tennis video - commentary generation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from PIL import Image
import time
import pyflow
import cv2
import numpy as np
import pandas as pd
import os

video_dir_path = "../data/videos/"
serve_file_path = "../data/annotations/serve.csv"
hit_file_path = "../data/annotations/hit.csv"
sdf = pd.read_csv(serve_file_path)
hdf = pd.read_csv(hit_file_path)
h_rows = hdf.shape[0]
s_rows = sdf.shape[0]
h_index = 0
s_index = 0
s_img_cnt = 0
h_img_cnt = 0
o_img_cnt = 0
vid_list = ["V007.mp4","V009.mp4","V008.mp4","V006.mp4","V0010.mp4"]

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


for vid in vid_list:
    vid_path = os.path.join(video_dir_path, vid)
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    prvs = image 
    count = 0
    h_event = False
    s_event = False
    o_event = False
    while success:
        if s_index < s_rows:
            curr_s_row = sdf.iloc[s_index]
        if h_index < h_rows:
            curr_h_row = hdf.iloc[h_index]
        s_start = curr_s_row["start"]
        h_start = curr_h_row["start"]
        s_end = curr_s_row["end"]
        h_end = curr_h_row["end"]
        h_side = curr_h_row["side"][0]
        h_player = curr_h_row["player"][0]
        s_player = curr_s_row["player"][0]

        if s_start<count and s_end>count:
            if not s_event:
                #prvs = image
                s_target_frame = count + 15
            s_event = True
        elif s_end == count:
            s_event = True
            s_index = s_index + 1
        else:
            s_event = False

        if h_start<count and h_end>count:
            if not h_event:
                #prvs = image
                h_target_frame = count + 10
            h_event = True
        elif h_end == count:
            h_event = True
            h_index = h_index + 1
        else:
            h_event = False

        if (not h_event) and (not s_event):
            if not o_event: 
                o_target_frame = count 
            o_event = True
        else:
            o_event = False
         
        if h_event:
            s_event = False
          #  o_event = False

        success,image = vidcap.read()
        if not success:
            break
        next = image
        if (o_event and o_target_frame == count and (not count == 0)): 
            print ("Processing video => ",vid)
            print ("Frame => ",count)
            next = cv2.resize( next , ( 224 , 224 ), interpolation = cv2.INTER_AREA )
            prvs = cv2.resize( prvs , ( 224 , 224 ), interpolation = cv2.INTER_AREA )
            #cv2.imwrite('1.jpg',prvs)
            #cv2.imwrite('2.jpg',next)
            #im1 = cv2.imread('1.jpg')
            #im2 = cv2.imread('2.jpg')
            im1 = np.array(prvs)
            im2 = np.array(next)
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.
            s = time.time()
            u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
            e = time.time()
            print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
                e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            #hsv = np.zeros(im1.shape, dtype=np.uint8)
            #hsv[:, :, 0] = 255
            #hsv[:, :, 1] = 255
            #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #hsv[..., 0] = ang * 180 / np.pi / 2
            #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
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
        prvs = next
    

        #if h_event and h_target_frame == count:
#            cv2.resize(flow_img, (512,512))
            #cv2.imwrite("../data/flow_dataset3/H%s%s/H_%s%s%d.png" % (h_player, h_side, h_player, h_side, h_img_cnt) , flow_img )     # save frame as JPEG file      
            #h_target_frame = h_target_frame + 10
            #x = cv2.imread("../data/flow_dataset3/H%s%s/H_%s%s%d.png" % (h_player, h_side, h_player, h_side, h_img_cnt))
            #y = cv2.resize(x,(512,512),interpolation = cv2.INTER_AREA)
            #cv2.imwrite("../data/flow_dataset3/H%s%s/H_%s%s%d.png" % (h_player, h_side, h_player, h_side, h_img_cnt) , flow_img )

            #h_img_cnt = h_img_cnt + 1

        #if s_event and s_target_frame == count:
#            cv2.resize(flow_img, (512,512))
            #cv2.imwrite("../data/flow_dataset3/S%s/S_%s%d.png" % (s_player, s_player, s_img_cnt) , flow_img )     # save frame as JPEG file      
            #x = cv2.imread("../data/flow_dataset3/S%s/S_%s%d.png" % (s_player, s_player, s_img_cnt))
            #y = cv2.resize(x,(512,512),interpolation = cv2.INTER_AREA)
            #cv2.imwrite("../data/flow_dataset3/S%s/S_%s%d.png" % (s_player, s_player, s_img_cnt) , y )
            #s_target_frame = s_target_frame + 15
            #s_img_cnt = s_img_cnt + 1

        if o_event and o_target_frame == count and (not count == 0):
            #flow_img = cv2.resize(flow_img, (224, 224), interpolation = cv2.INTER_AREA)
            cv2.imwrite("../data/flow_dataset/O/O_%d.png" % o_img_cnt , flow_img)     # save frame as JPEG file      
            #flow_img = cv2.imread("../data/flow_dataset/O/O_%d.png" % o_img_cnt)
            #flow_img = cv2.resize(flow_img, (224, 224), interpolation = cv2.INTER_AREA)
            #cv2.imwrite("../data/flow_dataset/O/O_%d.png" % o_img_cnt , flow_img)
            o_target_frame = o_target_frame + 200
            o_img_cnt = o_img_cnt + 1
        
        count += 1

cap.release()
cv2.destroyAllWindows()
