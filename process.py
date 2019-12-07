import os
import cv2
from src import detection as dt

CIRCLES_MODEL_PATH = 'models/detect_circles.joblib'
TRIANGLES_MODEL_PATH = 'models/detect_triangles.joblib'

cls_circles = dt.SVM(model_path=CIRCLES_MODEL_PATH)
cls_triangles = dt.SVM(model_path=TRIANGLES_MODEL_PATH)
