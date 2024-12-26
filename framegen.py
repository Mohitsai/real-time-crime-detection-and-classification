import av
import glob
import cv2
import os
import time
import datetime
import argparse
import numpy as np
def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0): 
        yield frame.to_image()
        
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
   
    

from tqdm.autonotebook import tqdm
path = r'D:\CAPSTONE\UCF-Anomaly-Detection-Dataset\UCF_Crimes\Videos'
result = r'D:\CAPSTONE\UCF-Anomaly-Detection-Dataset\UCF_Crimes\Videos\content\Dataset'

for i in tqdm(os.listdir(path)):
  if(i!='content'):
      print(i)
      p1 = os.path.join(path,i)
      r1 = os.path.join(result,i)
      if os.path.exists(r1):
                continue
      os.makedirs(r1,exist_ok = True)
      for j in os.listdir(p1):
                vid_path = os.path.join(p1,j)
                r2 = os.path.join(r1,j[:-4])
                os.makedirs(r2,exist_ok = True)
                for j, frame in enumerate((extract_frames(vid_path))):
                    frame.save(os.path.join(r2, f"{j}.jpg"))
    
path = r'D:\CAPSTONE\UCF-Anomaly-Detection-Dataset\UCF_Crimes\Videos\content\Dataset'
res = r'D:\CAPSTONE\UCF-Anomaly-Detection-Dataset\UCF_Crimes\Videos\content\FrameStrip'
seq_length = 32
dir = os.listdir(path)
for i in tqdm(dir):
      p1 = os.path.join(path,i)
      for j in os.listdir(p1):
          p2 = os.path.join(p1,j)
          l = 0
          skip_length = int(len(os.listdir(p2))/seq_length)
          for m in range(10):
              r3 =os.path.join(res,j.split('_')[0]+'_'+str(m)+'-'+i)
              os.makedirs(r3,exist_ok = True)
              k = m
              l=0
              while(l!=seq_length):
                  p3 = os.path.join(p2,str(k) + ".jpg")
                  img = cv2.imread(p3)
                  img = unsharp_mask(img)
                  #img = cv2.resize(img,(128,128))
                  r4 =os.path.join(r3 , str(l)+".jpg")
                  cv2.imwrite(r4,img)
                  k = k+skip_length
                  l = l+1
key= os.listdir(path)
print(key)
value =list (range(10))
crime = dict()
for i in key:
    if(i!='Normal_Videos_event'):
        crime[i]=1
    else:
        crime[i]=0
#crime = dict(zip(key,value))
file1 = open("train.txt", "w")
for j in os.listdir(res):
        file1.write(j + ' ' +'32' +' ' +str(crime[(j.split('-')[1])])+'\n')
file1.close()
