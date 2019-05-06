import pandas as pd
import os
import cv2
from pytube import YouTube

#download once and process many times
def download_and_process( video_save_path, row ):
    video_id = row['VideoID']
    file_name = row['file_name']
    path = os.path.join( video_save_path, file_name )
    if os.path.exists( path ):
        return -1

    start = row['Start']
    end = row['End']
    #path = os.path.join("/home/shashwatcs15/VideoAnalytics/",video_id)
    print "Saving video => https://www.youtube.com/watch?v=" + video_id
    if os.path.exists('tmp.mp4'):
        os.system('rm tmp.mp4')

    try:
        yt = YouTube("https://www.youtube.com/watch?v="+video_id)
    except:
        print "Error"
        return -1


   
    yt.streams.filter(file_extension='mp4',resolution='360p').first().download('.',filename='tmp')

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    cap = cv2.VideoCapture( 'tmp.mp4' )
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter( path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    start_frame = int(fps * start)
    end_frame = int(fps * end)
    frame_count = 0
    while frame_count < end_frame:
        ret, frame = cap.read()
        frame_count += 1

        if frame_count >= start_frame:
            out.write(frame)

    cap.release()
    out.release()



def main():
    #csv_path="./video_corpus.csv"
    csv_path = "./test_video_list.csv"
    video_save_path="./TestVideoDataset/"
    #video_save_path="./VideoDataset"	
    data = pd.read_csv(csv_path,sep=',')
    #data = data[data['Language']=='English']
    # add new column called file_name to each row (axis = 1)
    data['file_name']=data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi', axis = 1)
    #data.apply(lambda row: download_and_process(video_save_path,row),axis = 1)
    x = 0
    for i in range(len(data)):
        print "S.no. =================> "+str(i)
        if x == -1 and data.iloc[i]['file_name'] == data.iloc[i-1]['file_name']:
            continue
        x = download_and_process(video_save_path, data.iloc[i])


if __name__=="__main__":
    #download_and_process(None, None)
    main()
