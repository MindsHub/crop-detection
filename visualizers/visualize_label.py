import cv2
import numpy as np
import PIL.ImageOps
import PIL.Image

FILE = "./dataset/training/{}/FPWW0310081_RGB1_20200224_132013_6.png"

image = cv2.imread(FILE.format("images"), cv2.IMREAD_UNCHANGED)
cv2.imshow("Image", image)

label = cv2.imread(FILE.format("labels"), cv2.IMREAD_UNCHANGED)
cv2.imshow("Label", label)
cv2.imshow("Autocontrast", np.asarray(PIL.ImageOps.autocontrast(PIL.Image.fromarray(label))))

label *= 255
label = np.repeat(label.reshape(*label.shape, 1), 3, axis=2)
print(label.shape, image.shape)
image = cv2.addWeighted(label, 0.4, image, 0.6, 0)
cv2.imshow("Mixed", image)

while cv2.waitKey(1) & 0xFF != ord('q'): pass
cv2.destroyAllWindows()
