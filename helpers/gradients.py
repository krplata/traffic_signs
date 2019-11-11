import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

im = cv2.imread('./grad_example.png', cv2.IMREAD_COLOR)
gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

kernel_ox = np.array([-1, 0, 1])
kernel_oy = np.vstack(kernel_ox)

grad_x = cv2.filter2D(gr, -1, kernel_ox)
print(gr)
grad_y = cv2.filter2D(gr, -1, kernel_oy)
print(grad_x)
print(grad_y)
if np.array_equal(grad_x, grad_y):
    print("asdasd")
plt.figure()
plt.imshow(grad_x, cmap='gray')
plt.figure()
plt.imshow(grad_y, cmap='gray')
plt.show()