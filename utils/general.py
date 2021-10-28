import cv2
import os
import numpy as np

def readImg(path): # read image and change order of channel to RGB
	img = cv2.imread(path)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img

def load_ground_truth(dataset): # load ground truth of dataset
	path_gr = os.path.join(os.getcwd(), 'data', dataset, 'label_converted')
	ground_truth = dict()
	for file_gr in os.listdir(path_gr):
		path_file = os.path.join(path_gr,file_gr)
		with open(path_file,'r') as file:
			count = 0
			lines = file.readlines()
			for line in lines:
				if len(line) < 2:
					break
				if count == 0 :
					count += 1
					ground_truth[file_gr] = np.array([[float(x) for x in line.split('\n')[0].split(' ')]])
					continue
				ground_truth[file_gr] = np.append(ground_truth[file_gr],np.array([[float(x) for x in line.split('\n')[0].split(' ')]]), axis = 0)			
	return ground_truth

def load_predict(dataset, model): # load detection result of each model on a dataset
	path_pre = os.path.join(os.getcwd(), 'result', dataset, model, 'predict')
	predict = dict()
	for file_pre in os.listdir(path_pre):
		predict[file_pre] = np.array([])
		path_file = os.path.join(path_pre,file_pre)
		with open(path_file,'r') as file:
			count = 0
			lines = file.readlines()
			for line in lines:
				if count == 0:
					count += 1
					predict[file_pre] = np.array([[float(x) for x in line.split('\n')[0].split(' ')]])
					continue
				predict[file_pre] = np.append(predict[file_pre],np.array([[float(x) for x in line.split('\n')[0].split(' ')]]), axis=0)
	return predict

def write_result_mtcnn(img_name, bboxs, dataset): # write detection result of MTCNN
	result_file = img_name + '.txt'
	path_result = os.path.join(os.getcwd(), 'result', dataset , 'MTCNN',  'predict', result_file)
	with open(path_result, 'a+') as file:
		for bbox in bboxs:
			xmin = bbox['box'][0]
			ymin = bbox['box'][1]
			xmax = xmin + bbox['box'][2]
			ymax = ymin + bbox['box'][3]
			conf = round(bbox['confidence'],3)
			file.writelines('1 {} {} {} {} {}\n'.format(conf, xmin, ymin, xmax, ymax))

def write_result_scrfd(img_name,pr_box, dataset): # write detection result of SCRFD
  result_file = img_name + '.txt'
  path_result = os.path.join(os.getcwd(),'result', dataset, 'SCRFD', 'predict', result_file)
  with open(path_result,'a+') as file:
    for box in pr_box:
      bbox = np.asarray(box['bbox'])
      score = box['det_score']
      file.writelines('1 {} {} {} {} {}\n'.format(score, bbox[0], bbox[1], bbox[2], bbox[3]))

def write_result_retina(img_name,pr_box,dataset): # write detection result of RetinaFace
	result_file = img_name + '.txt'
	path_result = os.path.join(os.getcwd(), 'result', dataset , 'RetinaFace',  'predict', result_file)
	with open(path_result, 'a+') as file:
		for bbox in bboxs:
			xmin = bbox['facial_area'][0]
			ymin = bbox['facial_area'][1]
			xmax = xmin + bbox['facial_area'][2]
			ymax = ymin + bbox['facial_area'][3]
			conf = round(bbox['score'],3)
			file.writelines('1 {} {} {} {} {}\n'.format(conf, xmin, ymin, xmax, ymax))


