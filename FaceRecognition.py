import streamlit as st
import pickle
import cv2
import numpy as np
import numpy as np
import pandas as pd

st.markdown("<u><h1>Face Recogniton Technology</h1></u>",unsafe_allow_html=True)

def values(i):
    if i==0:
        return 'Shashank Purohit'
    elif i==1:
        return "Sakshi Nagpal"
    elif i==2:
        return "Sakshi Agarwal"
    elif i==3:
        return "Saksham Jain"
    


with open("faceAi.pickle",'rb') as f:
    lr=pickle.load(f)


face_cascade=cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt.xml")




img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img=cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)


    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    
    faces=face_cascade.detectMultiScale(img,1.1,4)

    for i ,(x,y,w,h) in enumerate(faces):
        #cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,255),2)

        cv2.rectangle(cv2_img,(x,y),(x+w,y+h),(0,255,255),2)

        face=cv2_img[y:y+h,x:x+w]
        face=cv2.resize(face,(28,28))
        img=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        x=np.array(img)/255
        st.image(x)
        x=x.flatten()
        x=x.reshape(1,-1)    
        
        st.write(values(lr.predict(x)))












