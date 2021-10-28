import sys
import os
sys.path.insert(1,os.path.join(os.getcwd()))
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from utils.general import write_result_scrfd
from utils.general import readImg, write_result_mtcnn, load_ground_truth, load_predict



datasets = {0:'PWMFD', 1:'MAFA'}
def scrfd_evaluate(data_id): # inference on dataset
	app = FaceAnalysis(allowed_modules=['detection']) # only enable detection
	app.prepare(ctx_id=0, det_size=(640, 640)) # set input size
	path_imgs = os.path.join(os.getcwd(),'data',datasets[data_id],'images')
	for file_name in os.listdir(path_imgs):
		path_img = os.path.join(path_imgs,file_name)
		img =  ins_get_image(path_img) # load image
		pr_box = app.get(img) # inferencing
		write_result_scrfd(file_name.split('.')[0], pr_box, datasets[data_id]) # write result to a .txt file
if __name__ == '__main__':
	for key, value in datasets:
		scrfd_evaluate(value)




