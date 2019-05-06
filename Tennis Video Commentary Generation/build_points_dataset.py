import pandas as pd
import os 
import cv2

def split_videos( video_dir, video_save_dir, video_csv_path ):
    split_info = pd.read_csv( video_csv_path, delimiter = "," )
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    prev_vid = "none"
    for index, row in split_info.iterrows():
        print "Point Id :: "+row["pid"]
        vid = row["vid"]+".mp4"
        subset = row["set"]
        if not subset == "val":
            continue
        if not prev_vid == vid:
            if not prev_vid == 'none' :
                cap.release()
            video_path = os.path.join( video_dir, vid )
            print "Capturing video :: "+ video_path
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            if int(major_ver)  < 3 :
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            else :
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = row["start"]
        end_frame = row["end"]
        s_path = os.path.join( video_save_dir, subset, vid+"_"+str(start_frame)+"_"+str(end_frame)+".avi")
        print "Saving clip :: "+s_path
        out = cv2.VideoWriter( s_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
        ret = True
        while frame_count < end_frame :
            ret, frame = cap.read()
            if frame_count >= start_frame and frame_count <= end_frame :
                out.write(frame)
            frame_count = frame_count + 1
        prev_vid = vid
        out.release()

def main():
    video_save_dir = '/home/shashwatcs15/VideoAnalytics/Tennis/data/points_data/'
    video_csv_path = '/home/shashwatcs15/VideoAnalytics/Tennis/data/splits/POINTS.csv'
    video_dir = '/home/shashwatcs15/VideoAnalytics/Tennis/data/videos/'
    split_videos( video_dir, video_save_dir, video_csv_path )

if __name__=='__main__':
    main()
