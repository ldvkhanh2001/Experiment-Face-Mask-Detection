import os
import sys
sys.path.insert(1,os.getcwd()) # add origin path to the system
from mtcnn.mtcnn import MTCNN
from utils.general import readImg, write_result_mtcnn, load_ground_truth, load_predict



datasets = {0:'PWMFD', 1:'MAFA'}
def mtcnn_evaluate(data_id):  # inference on dataset
	detector = MTCNN() # create MTCNN instance 
	path_imgs = os.path.join(os.getcwd(),'data',datasets[data_id],'images')
	for file_name in os.listdir(path_imgs):
		path_img = os.path.join(path_imgs,file_name)
		img = readImg(path_img)
		pr_bboxs = detector.detect_faces(img) # inferencing
		img_name = file_name.split('.')[0]
		write_result_mtcnn(img_name,pr_bboxs,datasets[data_id]) # write result to a .txt file

if __name__ == '__main__':
	for key, value in datasets.items():	
		mtcnn_evaluate(value)
	





