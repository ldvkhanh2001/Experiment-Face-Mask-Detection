import os
import sys
sys.path.insert(1, os.getcwd())
from utils.general import readImg, write_result_retina, load_ground_truth, load_predict
from retinaface import RetinaFace


datasets = {0:'PWMFD', 1:'MAFA'}
def retinaface_evaluate(data_id):   # inference on dataset
	path_imgs = os.path.join(os.getcwd(),'data',datasets[data_id],'images')
	for file_name in os.listdir(path_imgs):
		img_name = file_name.split('.')[0]
		path_img = os.path.join(path_imgs, file_name)
		img = readImg(path_img)
		pr_boxs = RetinaFace.detect_faces(img) # inferencing
		write_result_retina(img_name,pr_boxs,datasets[data_id]) #write result to a .txt file
if __name__ == '__main__':
	for key, value in datasets.items():
		retina_evaluate(value)

