from utils.general import load_ground_truth, load_predict
from utils.mAP_metric import mAP_score
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	datasets = ['PWMFD', 'MAFA']
	models = ['MTCNN', 'SCRFD', 'RetinaFace', 'Yolov5face']
	class_map = {1: 'face'}
	color = ['blue', 'red', 'green', 'yellow']
	
	mAP = mAP_score()
	for dataset in datasets:
		aps = []
		print('------------------------{}---------------------'.format(dataset))
		gr = load_ground_truth(dataset)
		for i in range(len(models)):
			pre = load_predict(dataset = dataset, model = models[i])
			pre, rec,ap = mAP.evaluate(pre, gr, dataset=dataset, model = models[i], class_map=class_map)
			index = []
			pre = np.asarray(pre)
			rec = np.asarray(rec)
			print('{} : {}%AP'.format(models[i], round(ap*100,2)))
			aps.append(round(ap*100,2))	
			plt.plot(rec,pre, linewidth = 3, color = color[i])
		plt.title('Precision - Recall Curve on {}'.format(dataset))
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.xticks(np.arange(0,1.2,0.2))
		#if dataset == 'PWMFD':
		plt.yticks(np.arange(0.6,1.0,0.1))
		plt.legend(['{} - {:.2f}%AP'.format(models[0], aps[0]), '{} - {:.2f}%AP'.format(models[1], aps[1]),
					 '{} - {:.2f}%AP'.format(models[2], aps[2]), '{} - {:.2f}%AP'.format(models[3], aps[3]) ])
		plt.show()
