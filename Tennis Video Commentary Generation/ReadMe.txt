:::::::::::::::::: Tennis Video Commentary Generation ::::::::::::::::::

Please refer the paper :: TenniSet: A Dataset for Dense Fine-Grained EventRecognition, Localisation and Description

1) Download the TenniSet Dataset

2) Usage of the files 
	
   a> create_event_csvs.py : Creates csv files for hit events (hit.csv) and serve events (serve.csv)
   
   b> create_points_csv.py : Creates csv files for point events (points_with_captions.csv)
   
   c> build_image_dataset.py : Used to sample images from the videos for different types of events

   d> build_points_dataset.py : Used to save videos of the points in .avi format

   e> extract_feature_vectors.py : Used to extract features of the RGB images using the trained CNN

   f> tennisVGGModel.py : custom VGG-16 model for our 7 class classifier 

   g> train.py : training the tennisVGGModel.py

   h> lstm-model.py : Lstm model to generate captions taking features of the RGB images as input

3) First generate the csv files using <a> and <b>.

4) Sample RGB images from the videos using <c> [We did not have a separate image dataset for different types of events. We sampled images from the videos itself.] All the images are saved locally into 7 different directories one for each class.

5) Split the videos into clips using the dataset using <d>. Each clip represents a point and has a commentary for it. All the point clips are stored locally. 

5) Train the cnn model on the sampled image dataset for classifying RGB images into 7 classes using <f> and <g>. 

6) Use the trained CNN model to extract features (prefinal layer of the CNN or the FC layer of the CNN) for any RGB image. In other words, we sample 80 images in a sequential order from the point-videos and for each image we obtain its representation using <e>. Each feature is 256 dimensional. We have 80 frames sampled from a point video. So for each point video we have 80 X 256 dimensional feature that gets saved locally as .npy representation of the video.

7) We can train lstm, using <h>, for representations from the videos (considering only RGB features or may also use the Optical flow representations alongwith RGB)

8) OPTICAL FLOW : 

In the pyflow (cloned from github) folder, Optical Flow is calculated using Brox Algorithm. 

--> Demo python files just as a tutorial on how to create optical flow between 2 images
--> brox_optical_flow.py - create flow image dataset from the videos
--> extract_flow_features.py - create flow feature representations similar to RGB, 80*256 dimensional for each video, using the trained Flow CNN.

9) Basic idea - We sample 80 frames from the video clip of a point event in Tennis. Each point event has a commentary. We fix each frame and for each frame we create a window around it [frame-4, frame+5] we calculate optical flow between these two frames. Now, corrosponding to every RGB central frame, we have an optical flow image representing the temporal changes in that window. LSTM has 80 timestamps - at each timestep we give an input vector to the LSTM, this input vector is made up of 256 dimensional rgb representation and 256 dimensional flow representation of the window around the fixed rb frame. Thus, 512 dimensional input at each timestep to the LSTM input.
Also, with certain modifications we can train the LSTM only on the RGB representations by making the LSTM input to 256 rather than 512. 

10) Evaluation : 
In the main function of lstm_model.py comment train(). Uncomment test(). Provide the correct paths to the test video features. (Note : test video rgb and flow features must have been computed and saved.)  For all the videos, captions are generated and saved in the file - S2VT_results.txt and the actual captions are stored in actual_captions.txt. 

In the cococaption(cloned from github), like in open domain video captioning, run the file eval_tennis.py to get the METEOR, BLEU, ROUGEL scores.  