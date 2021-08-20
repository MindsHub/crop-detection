import cv2
import numpy as np
import PIL.ImageOps
import PIL.Image

FILE = "./dataset/test/{}/A_000504.png"

image = cv2.imread(FILE.format("images"), cv2.IMREAD_UNCHANGED)
cv2.imshow("Image", image)

label = cv2.imread(FILE.format("labels"), cv2.IMREAD_UNCHANGED)
cv2.imshow("Label", label)
cv2.imshow("Autocontrast", np.asarray(PIL.ImageOps.autocontrast(PIL.Image.fromarray(label))))

label *= 128
image[:,:,0] += label
image[:,:,1] += label
image[:,:,2] += label
cv2.imshow("Mixed", image)

while cv2.waitKey(1) & 0xFF != ord('q'): pass
cv2.destroyAllWindows()
