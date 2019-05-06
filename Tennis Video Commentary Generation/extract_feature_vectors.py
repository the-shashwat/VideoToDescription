import numpy as np
import gc
import os,sys

# specify gpu config
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "7"
session = tf.Session(config=config)
import cv2
import skimage
from keras.backend.tensorflow_backend import set_session
set_session(session)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from tennisVGGModel import TennisVGGNet
from keras import backend as K
import numpy as np

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


def extract_features(img , model):
    features = model.predict(img)
    print "after model.predict"
    print features.shape
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    print "After squeeze"
    print features.shape
    return features

def main():
    frames_to_sample = 80
    print ("frames to sample from each video = "+str(frames_to_sample))
    video_dir = './data/points_data/train/' 
    #video_save_dir = './Features'
    video_save_dir = './data/points_data/train_flow_feats2'     # features are stored here 
    videos = os.listdir(video_dir)
    videos = filter(lambda x:x.endswith('avi'),videos)
    model = TennisVGGNet.build(width=image_dims[1], height=image_dims[0],
        depth=image_dims[2], classes=7)
    model.load_weights("./flow-cnn-models/weights-improvement-69-0.59.hdf5")

    intermediate_layer_model = Model(inputs=model.input, 
                                 outputs=model.get_layer("fc").output)
    for index,video in enumerate(videos):
	    video = videos[index]
        print str(index)+" -----> "+str(video)
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

        # if frame count is more than 80 then extract out 80 frames using a linear spacing
        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=frames_to_sample, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        # uncomment this for rgb
        # cropped_frame_list = np.array(map(lambda x: preprocess_frames(x), frame_list))
      
        cropped_frame_list = np.array(map(lambda x: preprocess_frames_of(x), frame_list))
        of = np.zeros(np.shape(cropped_frame_list), dtype = int)
        for idx,frame_id in enumerate(frame_indices):
            start = frame_id - 3 
            end = frame_id + 2
            if start < 1 :
                start = 1
            if end >= frame_count:
                end = frame_count - 1
            next = cropped_frame_list[ end ]
            prvs = cropped_frame_list[ start ]
            of[idx] = calc_OF( prvs, next )

        
        print cropped_frame_list.shape
        print of.shape
	    #cropped_frame_list *= 255 
        #cv2.imwrite('after.png',cropped_frame_list[70])
        feats = np.zeros([len(cropped_frame_list)]+[256])
	    batch_size = 16
	    for start,end in zip(range(0,len(cropped_frame_list)+batch_size,batch_size),range(batch_size,len(cropped_frame_list)+batch_size,batch_size)):
		    feats[start:end] = extract_features(cropped_frame_list[start:end],intermediate_layer_model)
        print feats
	    print feats.shape
	# feats has dimension of (number of frames, 4096)
        # feats = cnn.get_features(cropped_frame_list)

       	save_full_path = os.path.join(video_save_dir, video + '.npy')
        np.save(save_full_path, feats)
        #del feats
	#gc.collect()

def calc_OF( prvs, next, idx, video, i ):
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
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
    return flow_img

def preprocess_frames_of(frame):
    frame = cv2.resize(frame, (512,512), interpolation = cv2.INTER_AREA)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return frame

if __name__=="__main__":
    main()



