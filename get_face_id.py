import os

import numpy as np

import cv2
from math import sqrt


class face_id_getter:

    def __init__(self, targetId=0, FP=32):
        self.FP = FP
        self.weights = 'face-reidentification-retail-0001/FP{}/face-reidentification-retail-0001.bin'.format(self.FP)
        self.config = 'face-reidentification-retail-0001/FP{}/face-reidentification-retail-0001.xml'.format(self.FP)
        self.framework = 'DLDT'
        self.targetId = targetId
        self.model = cv2.dnn.readNet(self.weights, self.config, self.framework)
        self.model.setPreferableTarget(targetId=self.targetId)

    def get_answer(self, input_data):
        blob = cv2.dnn.blobFromImage(input_data, size=(128, 128), crop=False)
        self.model.setInput(blob)
        ans = self.model.forward()[0, : , 0, 0]
        length = 0
        for i in ans:
            length += i * i
        length = sqrt(length)
        ans = np.array(ans)
        ans /= length
        return ans


if __name__ == '__main__':
    getter = face_id_getter()
    dirs = list(os.walk('faces'))[1:]
    paths = []
    for dir_ in dirs:

        for _t in dir_[2]:
            paths.append(dir_[0] + '\\' +  _t)

    with open('labels.txt', 'w') as f:

        for path in paths:
            data = cv2.imread(path)
            vec = getter.get_answer(data)
            
            f.write(str(path[6:7]) + ' : ' + str(' '.join(list(map(str, vec)))) + '\n')