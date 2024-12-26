import av
import glob
import cv2
import os
import time
import tqdm
import argparse
import numpy as np
import argparse
import datetime
import imutils
from PIL import Image as im
from imutils.video import VideoStream
from tqdm.autonotebook import tqdm
import shutil

def motionDetectCamera():
        print("Starting Camera...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        firstFrame = None
        min_area=2000
        frrate=0
        while True:
            frame = vs.read()
            temp=frame
            text = "No motion Detected"
            if frame is None:
                break
            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if firstFrame is None:
                firstFrame = gray
                continue
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 15, 245, cv2.THRESH_BINARY)[1] #threshold value to be fine tuned before presentation
            thresh = cv2.dilate(thresh, None, iterations=10)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) < min_area :
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Motion detected!"	
            cv2.putText(frame, "Motion detection status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            cv2.imshow("Input Footage", frame)
            cv2.imshow("Detection Threshold", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(1) & 0xFF
            if text=="Motion detected!":
                frrate=frrate+1
                if(frrate==18):
                    yield im.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
                    frrate=0
            if key == ord("q"):
                break
        vs.stop()
        cv2.destroyAllWindows()

def motionDetectVideos(videos):
        print("Processing videos from path...")
        vs = cv2.VideoCapture(videos)
        firstFrame = None
        min_area=2000
        while True:
            frame = vs.read()
            frame = frame[1]
            text = "No motion Detected"
            if frame is None: #end of video
                break
            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if firstFrame is None:
                firstFrame = gray
                continue
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY)[1] #threshold value to be fine tuned before presentation
            thresh = cv2.dilate(thresh, None, iterations=10)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Motion detected!"	
            cv2.putText(frame, "Motion detection status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            cv2.imshow("Input Footage", frame)
            cv2.imshow("Detection Threshold", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(1) & 0xFF
            if text=="Motion detected!":
                yield im.fromarray(frame)
            if key == ord("Q"):
                break
        vs.release()
        cv2.destroyAllWindows()
        
def sharpenFrames(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


#driver code
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode",default="0", help="enter 0 for camera, 1 for path")
ap.add_argument("-v", "--video",default=None, help="enter path to the video folder")
ap.add_argument("-o", "--output",default=None, help="enter frames output path")
ap.add_argument("-f", "--frame",default=None, help="enter framestrip output path")
args = vars(ap.parse_args())
if args["mode"]=="0":
    op=args["output"]
    fr=args["frame"]
    if not os.path.exists(op):
        os.makedirs(op,exist_ok = True)
    if not os.path.exists(fr):
        os.makedirs(fr,exist_ok = True)
    for j, frame in enumerate((motionDetectCamera())):
        frame.save(os.path.join(op, f"{j}.jpg"))    
    seq_length = 18    
    k = 0
    l = 0
    for i in os.listdir(op):
        p1 = os.path.join(op,i)
        skip_length = int(len(os.listdir(op))/seq_length)     
        while(l!=seq_length):
            p2 = os.path.join(op,str(k) + ".jpg")
            img = cv2.imread(p2)
            img = sharpenFrames(img)
            img = cv2.resize(img,(480,360))
            if(k==0):
                img1 = img
            else:
                img1 = np.append(img1,img,axis = 1)
            k = k+skip_length
            l = l+1    
        cv2.imwrite(os. path. join(fr , str(k)+".jpg") ,img1)
    shutil.rmtree(op)

  
elif args["mode"]=="1": 
    path=args["video"]   
    result=args["output"]
    res=args["frame"]
    for i in tqdm(os.listdir(path)):
        p1 = os.path.join(path,i)
        r1 = os.path.join(result,i)
        if os.path.exists(r1):
                    continue
        os.makedirs(r1,exist_ok = True)
        for j in os.listdir(p1):
            vid_path = os.path.join(p1,j)
            r2 = os.path.join(r1,j[:-4])
            os.makedirs(r2,exist_ok = True)
            for j, frame in enumerate((motionDetectVideos(vid_path))):
                frame.save(os.path.join(r2, f"{j}.jpg"))
        
    path = result
    seq_length = 18
    dir = os.listdir(path)
    for i in tqdm(dir):
        p1 = os.path.join(path,i)
        r1 = os.path.join(res,i)
        os.makedirs(r1,exist_ok = True)
        for j in os.listdir(p1):
            p2 = os.path.join(p1,j)
            r2 = os.path.join(r1,j)
            l = 0
            skip_length = int(len(os.listdir(p2))/seq_length)
            for m in range(10):
                k = m
                while(l!=seq_length):
                    p3 = os.path.join(p2,str(k) + ".jpg")
                    img = cv2.imread(p3)
                    img = sharpenFrames(img)
                    img = cv2.resize(img,(480,360))
                    if(k==0):
                        img1 = img
                    else:
                        img1 = np.append(img1,img,axis = 1)
                    k = k+skip_length
                    l = l+1    
                cv2.imwrite(r2 + str(m)+".jpg",img1)
  
                      