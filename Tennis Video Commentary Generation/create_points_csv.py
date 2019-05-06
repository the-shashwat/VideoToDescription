import os
import pandas as pd

fd = open("./data/points_data/points_with_captions.csv","w")
fd.write("pid,vid,set,caption")
fd.write("\n")
points = pd.read_csv("./data/splits/POINTS.csv")

captions = {}
with open("./data/commentary/sents_train.txt","r") as f:
    for line in f.readlines():
        key, value = line.split("\t")
        captions[key] = value

with open("./data/commentary/sents_test.txt","r") as f:
    for line in f.readlines():
        key, value = line.split("\t")
        captions[key] = value

with open("./data/commentary/sents_val.txt","r") as f:
    for line in f.readlines():
        key, value = line.split("\t")
        captions[key] = value

#video_feat_dir = "/home/shashwatcs15/VideoAnalytics/Tennis/data/points_data/"

for index, row in points.iterrows():
    start_frame = row["start"]
    end_frame = row["end"]
    subset = row["set"]
    vid = row[ "vid" ] + ".mp4_" + str( start_frame ) + "_" + str( end_frame ) + ".avi.npy" 
    fd.write( row[ "pid" ] + "," + vid  + "," + subset + "," + captions[ row[ "pid" ] ] ) 

fd.close()
