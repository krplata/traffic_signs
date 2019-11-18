import Augmentor
from Augmentor import Pipeline as augpipe
import cv2

p = augpipe(source_directory='example')
p.shear(1, 20, 20)
p.process()