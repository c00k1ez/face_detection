import cv2
import numpy as np
import time
import os

print('скрипт делает 3 фотографии')
cap = cv2.VideoCapture(0)

FP = 32
targetId = 0
weights = 'face-detection-adas-0001/FP{}/face-detection-adas-0001.bin'.format(FP)
config = 'face-detection-adas-0001/FP{}/face-detection-adas-0001.xml'.format(FP)
framework = 'DLDT'

model = cv2.dnn.readNet(weights, config, framework)
model.setPreferableTarget(targetId=targetId)

with open('count.txt', 'r') as f: 
    cnt = int(f.read())

if not os.path.exists("faces/{}".format(cnt)):
    os.makedirs("faces/{}".format(cnt))

c = 0

c_f = 0

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), crop=False)
    print('для сохранения фото нажмите англ. а ')
    model.setInput(blob)
    ans = model.forward()
    for i in range(0, 200):
        if ans[0, 0, i, 2] > 0.5:
            x_min, y_min, x_max, y_max = np.array(ans[0, 0, i, 3:7]) * np.array([640, 480, 640, 480])
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), ( 0, 255, 255))
        
    if cv2.waitKey(1) & 0xFF == ord('a'):
        name = "faces/{}/{}_{}.jpg".format(cnt, cnt, c)
        cv2.imwrite(name, frame[int(y_min):int(y_max), int(x_min):int(x_max)])
        print('сохранено : ' + name)
        c += 1

    if c == 3:
        print('уряяяяяя')
        break
    
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cnt += 1

with open('count.txt', 'w') as f:
    f.write(str(cnt))

cap.release()
cv2.destroyAllWindows() 