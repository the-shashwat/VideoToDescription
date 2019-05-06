:::::::::::::::::: Open Domain Video Captioning ::::::::::::::::::

1) Download the Youtube videos in video_corpus.csv using downloadDataset.py

2) preprocess_videos.py - It contains the VGG-16 net pretrained on imagenet dataset. For all the videos it samples 80 frames in a linearly spaced manner from the video and for each frame we get a 4096 dimensional vector representation. Thus, for the video we have 80 x 4096 dimensioanl representation. This is saved as a .npy file for each video.

3) model.py - It contains the LSTM model for generating the captions.