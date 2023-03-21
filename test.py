import os
import pickle
import cv2

import numpy as np
import pandas as pd


def predict_value(val):
    if val==0:
        return 'shashank'
    elif val==1:
        return 'sn'
    elif val==2:
        return 'sa'
    elif val==3:
        return 'saksham'

def fun(folder_name): 
  n=1
  dataset=[]
  for i in os.listdir(f"{folder_name}/"):
    if ".jpg" in i:
        img=cv2.imread(f"{folder_name}/{i}")
        img=cv2.resize(img,(28,28))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x=np.array(img)/255
        dataset.append(x.flatten())
        
        print(f" converted {folder_name} : - {i} -- {n}")
        n+=1
  return dataset


with open('faceAi.pickle','rb') as f:
            lr=pickle.load(f)

s=fun("photos")
d=pd.DataFrame(s)

print(lr.predict(s))