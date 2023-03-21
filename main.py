import os
import cv2

user=input("enter the name of the user")


#data Generation part

if  not os.path.exists(f"{user}"):
    os.mkdir(f"{user}")


face_cascade=cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt.xml")

cap=cv2.VideoCapture(0)

n=0
while(n<10):
    ret,vid=cap.read()
    # vid=vid[0:, 200 : 1000]
    vid=vid[0:, 150 : 900]
    img=cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)

    
    faces=face_cascade.detectMultiScale(img,1.1,4)
    if len(faces)>0:
        print("detected")
    else:
        print("------ NOT DETECTED -----")

    for i ,(x,y,w,h) in enumerate(faces):
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,255),2)
        face=vid[y:y+h,x:x+w]
        

        cv2.imshow(f"face",face)
        cv2.imwrite(f"{user}/{user[0].capitalize()}_{n}.jpg",face)


    n+=1

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
