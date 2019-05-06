# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import pyflow
import numpy as np
import gc
import os,sys
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "1"
session = tf.Session(config=config)
import cv2
import skimage
from keras.backend.tensorflow_backend import set_session
set_session(session)
#from keras.utils import multi_gpu_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from tennisVGGModel import TennisVGGNet
from keras import backend as K
import numpy as np
#from extract_features import *

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

image_dims = (224,224,3)
# we need to preprocess the frames in the video before feeding them into the cnn
def preprocess_frames(image, req_width = 224, req_height = 224):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None],3)
    if len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape

    if width==height:
        resized_image = cv2.resize(image, (req_height,req_width))

    elif height < width:

        # first scale out the image to height = (w/h)*req_height and width = req_width
        resized_image = cv2.resize(image, (int(width * float(req_height) / height), req_width))

        # now find the cropping length
        # since cropping is done from both the ends so cropping length is divided by 2
        cropping_length = int((resized_image.shape[1] - req_height) / 2)

        # crop the image's height from top and bottom
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:

        # first scale the image to height = req_height and width = (h/w)*req_width
        resized_image = cv2.resize(image, (req_height, int(height * float(req_width) / width)))

        # now find the cropping length
        # since cropping is done from both the ends so cropping length is divided by 2
        cropping_length = int((resized_image.shape[0] - req_width) / 2)

        # crop the images width from left and right
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    return cv2.resize(resized_image, (req_height, req_width))

'''def extract_features(img):
    #with tf.device("/gpu:2"):
    base_model = VGG16(weights='imagenet', pooling=max, include_top = True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    #image = img.reshape((img.shape[0], img.shape[1], img.shape[2], img.shape[2]))
    image = preprocess_input(img)
    features = model.predict(image)
    print features.shape
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    print features.shape
    print features
    del model
    gc.collect()
    return features
'''
def extract_features(img , model):
    #with tf.device("/gpu:2"):
    #base_model = VGG16(weights='imagenet', pooling=max, include_top = True)
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
 #   model = multi_gpu_model(model, gpus=2)
    #img = image.load_img('./dog3.jpg', target_size=(224, 224))
    #print "after loading image"
    #print img
    #print img.shape
    '''img = np.array(list(map(lambda x:img_to_array(x), img)))
    print "After img to array"
    print img
    print img.shape
    print "After expand dims"'''
    #img = np.expand_dims(img, axis=0)
    #print img
    #print img.shape
    #img = preprocess_input(img)
    
    #print "after preprocessing"
    #print img
    #print img.shape
    #features = get_prefinal_layer_output([img, 0])[0]
    features = model.predict(img)
    #print np.shape(P)
    print ("after model.predict")
    print (features.shape)
#    print features
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    #np.squeeze(features)
    print ("After squeeze")
    print (features.shape)
#    print features
    #K.clear_session()
    #del model
    #gc.collect()
    '''for x,row in enumerate(P):
        print "frame "+str(x)
    	for (i, (imagenetID, label, prob)) in enumerate(P[x]):
        	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))   
    	print "----------------------"'''
    return features

def main():
    frames_to_sample = 80
    print ("Frames to sample from each video = "+str(frames_to_sample))
    video_dir = '../data/points_data/test/'
    #video_save_dir = './Features'
    video_save_dir = '../data/points_data/test_flow_feats2/' 
    videos = os.listdir(video_dir)
    videos = filter(lambda x:x.endswith('avi'),videos)
    model = TennisVGGNet.build(width=image_dims[1], height=image_dims[0],
        depth=image_dims[2], classes=7)
    #cnn =  CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)
    # prev_cnn_models was used for rgb pipeline
    model.load_weights("../flow-cnn-models4/weights-improvement-99-0.67.hdf5")

    #get_prefinal_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                              [model.layers[-2].output])
    intermediate_layer_model = Model(inputs=model.input, 
                                 outputs=model.get_layer("fc").output)
    #index = vid
    for index,video in enumerate(videos):
	video = videos[index]
        print (str(index)+" -----> "+str(video))
	index += 1
#        if os.path.exists(os.path.join(video_save_dir, video + '.npy')):
#            continue
        
	video_path = os.path.join(video_dir, video)
        try:
            cap = cv2.VideoCapture( video_path )
        except:
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1

        frame_list  = np.array(frame_list)
        size = len(frame_list)
        # if frame count is more than 80 then extract out 80 frames using a linear spacing
        if frame_count > frames_to_sample:
            frame_indices = np.linspace(0, frame_count, num=frames_to_sample, endpoint=False).astype(int)
            size = frames_to_sample
        else:
            frame_indices = np.linspace(0, frame_count, num=frame_count, endpoint=False).astype(int)
        #    frame_list = frame_list[frame_indices]

        # uncomment this for rgb
        # cropped_frame_list = np.array(map(lambda x: preprocess_frames(x), frame_list))
      
        of = np.zeros( ( size, 224, 224, 3 ) , dtype = float)
        print ("Sampling flow images for the clip...")
        for idx,frame_id in enumerate(frame_indices):
            start = frame_id - 3 
            end = frame_id + 2
            if start < 1 :
                start = 1
            if end >= frame_count:
                end = frame_count - 1
            next = frame_list[ end ]
            prvs = frame_list[ start ]
            of[idx] = calc_OF( prvs, next )
            #print (of[idx].shape)        
            #print (of[idx])
        of /= 255.0
        # before extracting the features from the frames we need to preprocess the frames
        #cv2.imwrite('b4.png',frame_list[70])
        #cropped_frame_list = np.array(map(lambda x: preprocess_frames(x), frame_list))
#        cropped_frame_list *= 255

#        for idx,frame in enumerate(cropped_frame_list):
 #           print frame
  #          print "###############"
   #         cv2.imwrite("./resized-images/frame_"+str(idx)+".png",frame)
    #    break
	#print cropped_frame_list.shape
        #cropped_frame_list = np.array(frame_list)
	#cropped_frame_list *= 255 
        #cv2.imwrite('after.png',cropped_frame_list[70])
        feats = np.zeros([len(of)]+[256])
	batch_size = 16
	for start,end in zip(range(0,len(of)+batch_size,batch_size),range(batch_size,len(of)+batch_size,batch_size)):
		feats[start:end] = extract_features(of[start:end],intermediate_layer_model)
        print (feats)
	print (feats.shape)
	# feats has dimension of (number of frames, 4096)
        # feats = cnn.get_features(cropped_frame_list)

       	save_full_path = os.path.join(video_save_dir, video + '.npy')
        np.save(save_full_path, feats)
        #del feats
	#gc.collect()

def calc_OF( prvs, next ):
    next = cv2.resize( next , ( 224 , 224 ), interpolation = cv2.INTER_AREA )
    prvs = cv2.resize( prvs , ( 224 , 224 ), interpolation = cv2.INTER_AREA )
    prvs = np.array(prvs, dtype = float) / 255.0
    next = np.array(next, dtype = float) / 255.0
    u, v, im2W = pyflow.coarse2fine_flow(
    prvs, next, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

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
#    cv2.imwrite(os.path.join("temp.png"),flow_img)
    #x = cv2.imread("temp.png")
    #y = cv2.resize(x,(224,224),interpolation = cv2.INTER_AREA)
    return flow_img

if __name__=="__main__":
#     img = cv2.imread('./opticalhsv.png')
#     img = preprocess_frames(img)
#     cv2.imwrite('preprocessed.png',img)
    main()



