import cv2
import os

extern_path = './data/recognition/external'
bw_path = './data/recognition/extern_bw'
for filename in os.listdir(extern_path):
    if filename.endswith('.jpg'):
        src_path = os.path.join(extern_path, filename)
        image = cv2.imread(src_path, cv2.IMREAD_COLOR)
        #image = reshape.im_prepare(image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(bw_path, filename)
        cv2.imwrite(output_path, grey)