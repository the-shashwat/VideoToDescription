import numpy as np
import gc
import os,sys

# specify the gpu requirements
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "2"
session = tf.Session(config=config)


import cv2
import skimage
from keras.backend.tensorflow_backend import set_session
set_session(session)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras import backend as K
import numpy as np

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
    img = preprocess_input(img)    
    #print "after preprocessing"
    #print img
    #print img.shape
    features = model.predict(img)
    #print "after model.predict"
    #print features.shape
    #print features
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    #np.squeeze(features)
    #print "After squeeze"
    #print features.shape
    #print features
    return features

def main():
    # as per the paper we have to sample 80 frames from each video
    #vid = int(sys.argv[1]) 
    frames_to_sample = 80
    print ("frames to sample from each video = "+str(frames_to_sample))
    video_dir = './VideoDataset'
    video_save_dir = './Features' 
    videos = os.listdir(video_dir)
    videos = filter(lambda x:x.endswith('avi'),videos)

    # load vgg16 model pretrained on imagenet dataset
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    for index,video in enumerate(videos):
	video = videos[index]
        print str(index)+" -----> "+str(video)
	index += 1
        if os.path.exists(os.path.join(video_save_dir, video + '.npy')):
            continue
        
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

        # before extracting the features from the frames we need to preprocess the frames
        #cv2.imwrite('b4.png',frame_list[70])
        cropped_frame_list = np.array(map(lambda x: preprocess_frames(x), frame_list))

	    #print cropped_frame_list.shape
        #cropped_frame_list = np.array(frame_list)
        print cropped_frame_list.shape
	    cropped_frame_list *= 255 
        feats = np.zeros([len(cropped_frame_list)]+[4096])
	    batch_size = 16
	    for start,end in zip(range(0,len(cropped_frame_list)+batch_size,batch_size),range(batch_size,len(cropped_frame_list)+batch_size,batch_size)):
		    feats[start:end] = extract_features(cropped_frame_list[start:end],model)
            print feats
         	print feats.shape
	    # feats has dimension of (number of frames, 4096)

       	save_full_path = os.path.join(video_save_dir, video + '.npy')
        np.save(save_full_path, feats)
       

if __name__=="__main__":
    main()
