import tensorflow
from tensorflow import keras
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import os
import cv2
import numpy as np
import time 
import random
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

import random 
msg=''
while True:
    _, frame = cap.read()
    #time.sleep(0.5)
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            msg=''
            url=''
            if label=='Happy' or label=='Surprise':
                happy_emoji = '\U0001F604'
                #msg='stress level 0 '+str(random.randint(0,10))+'%'+'\nyour are happy no need to worry'
                msg = f'\nstress level 0 {random.randint(0, 10)}% \n\n{happy_emoji}\n\nyou are happy, no need to worry'
                #print("texttt")
            if label=='Neutral' or label=='Angry':
                msg='stress level 1 '+str(random.randint(30,50))+'%'
                url='https://youtu.be/YoSuVws4OTQ?si=lDY5ICOI1UQWYq78'
            if label=='Disgust' or label=='Fear':
                msg='stress level 2 '+str(random.randint(60,80))+'%'
                url='https://youtu.be/lHVYgnlukTw?si=9I5rh1f-lTvnvaCe'
            cv2.putText(frame,msg,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import webbrowser   
# then make a url variable 
if url !='':  
# then call the default open method described above 
    webbrowser.open(url)

print(msg)
# show message box 
from tkinter import * 
from tkinter import messagebox 
  
root = Tk() 
root.geometry("400x400") 
  
w = Label(root, text =msg, font = "50")  
w.pack() 

root.mainloop()  

# show url - process launch browser
