import cv2
import pandas as pd
import os

video_dir_path = "./data/videos/"
serve_file_path = "./data/annotations/serve.csv"
hit_file_path = "./data/annotations/hit.csv"
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

for vid in vid_list:
    print "Processing video => "+vid
    vid_path = os.path.join(video_dir_path, vid)
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    count = 0
    h_event = False
    s_event = False
    o_event = False
    while success:
        print count
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

        if h_start<count and h_end>count:
            if not h_event:
                h_target_frame = count
            h_event = True
        elif h_end == count:
            h_event = True
            h_index = h_index + 1
        else:
            h_event = False

        if s_start<count and s_end>count:
            if not s_event:
                s_target_frame = count + 5 
            s_event = True
        elif s_end == count:
            s_event = True
            s_index = s_index + 1
        else:
            s_event = False
        
        if (not h_event) and (not s_event):
            if not o_event: 
                o_target_frame = count 
            o_event = True
        else:
            o_event = False 

        if h_event and h_target_frame == count:
            cv2.imwrite("./data/hit_images/H_%s%s%d.jpg" % (h_player, h_side, h_img_cnt) , image)     # save frame as JPEG file      
            h_target_frame = h_target_frame + 10
            h_img_cnt = h_img_cnt + 1

        if s_event and s_target_frame == count:
            cv2.imwrite("./data/serve_images/S_%s%d.jpg" % (s_player, s_img_cnt) , image)     # save frame as JPEG file      
            s_target_frame = s_target_frame + 10
            s_img_cnt = s_img_cnt + 1

        if o_event and o_target_frame == count:
            cv2.imwrite("./data/other_images/O_%d.jpg" % o_img_cnt, image)     # save frame as JPEG file      
            o_target_frame = o_target_frame + 100
            o_img_cnt = o_img_cnt + 1
        
        success,image = vidcap.read()
        count += 1
