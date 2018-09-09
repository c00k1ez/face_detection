import numpy as np
import cv2
import sys

from get_face_id import face_id_getter
from check import check
from win10toast import ToastNotifier 

FP = 32

targetId = 0

toaster = None

if '-use_notifications' in sys.argv:
    toaster = ToastNotifier() 

if len(sys.argv) > 1 and '-m' in sys.argv:
    FP =  16
    targetId = cv2.dnn.DNN_TARGET_MYRIAD



cap = cv2.VideoCapture(0)

getter = None

if '-get_face_id' in sys.argv:
    getter = face_id_getter()

weights = 'face-detection-adas-0001/FP{}/face-detection-adas-0001.bin'.format(FP)
config = 'face-detection-adas-0001/FP{}/face-detection-adas-0001.xml'.format(FP)

weights_emotions, config_emotions, emotions = None, None, None



if len(sys.argv) > 1 and '-use_emotions' in sys.argv:
    weights_emotions = 'emotions-recognition-retail-0003/FP{}/emotions-recognition-retail-0003.bin'.format(FP)
    config_emotions = 'emotions-recognition-retail-0003/FP{}/emotions-recognition-retail-0003.xml'.format(FP)
framework = 'DLDT'

model = cv2.dnn.readNet(weights, config, framework)
model.setPreferableTarget(targetId=targetId)

if len(sys.argv) > 1 and '-use_emotions' in sys.argv:
    emotions = cv2.dnn.readNet(weights_emotions, config_emotions, framework)
    emotions.setPreferableTarget(targetId=targetId)

emotions_decode = ('neutral', 'happy', 'sad', 'surprise', 'anger')

names = ["Plotnikov Egor", "Vainberg Roman", "Sataev Emil", "Unknown person"]

emotion_text = None

while(True):
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), crop=False)

    have_nots = False

    model.setInput(blob)
    ans = model.forward()
    for i in range(0, 200):
        x_min, y_min, x_max, y_max = np.array(ans[0, 0, i, 3:7]) * np.array([640, 480, 640, 480])
        if ans[0, 0, i, 2] > 0.5:
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), ( 0, 255, 255))
            
            if len(sys.argv) > 1 and '-use_emotions' in sys.argv:
                blob_emotions = cv2.dnn.blobFromImage(frame[int(y_min):int(y_max), int(x_min):int(x_max)], size=(64, 64), crop=False)
                emotions.setInput(blob_emotions)
                ans_emotions = emotions.forward()[0, : , 0 , 0]
                ans_emotions = list(map(lambda x: 1 if x > 0.5 else 0, ans_emotions))
                _t = ''.join(list(map(str,ans_emotions))).find('1')
                if _t == -1:
                    _t = 0
                emotion_text = emotions_decode[_t]

            if '-get_face_id' in sys.argv:
                _ans = getter.get_answer(frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                
                t = check('labels.txt', _ans)
                #print(names[t])
                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame,names[t],(int(x_min), int(y_min)), font, 1,(255,255,255),2,cv2.LINE_AA)
                if emotion_text != None:
                    cv2.putText(frame,emotion_text,(int(x_min), int(y_max)), font, 1,(255,255,255),2,cv2.LINE_AA)

            if '-use_notifications' in sys.argv and not have_nots:
                toaster.show_toast("Welcome, " + names[t],"")
                have_nots = True

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows() 