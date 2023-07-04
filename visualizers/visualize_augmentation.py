import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import sys

sys.path = ['./training'] + sys.path
from dataset import getDataset

batch_size = 16
d = getDataset(True, batch_size)
for images, labels in d:
    break

h, w, c = images.shape[1:]
res = np.zeros((h * 4, w * (batch_size // 2), c), dtype=np.float32)

for b in range(batch_size // 2):
    y = w * b
    for i, x in [(b, 0), (batch_size // 2 + b, h * 2)]:
        image, label = images[i], labels[i]
        res[x:x+h, y:y+w] = image / 255
        x += h
        res[x:x+h, y:y+w] = label

cv2.namedWindow("Data", cv2.WINDOW_NORMAL)
cv2.imshow("Data", res)
while cv2.waitKey(1) & 0xFF != ord('q'): pass
cv2.destroyAllWindows()