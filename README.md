# Experiment-Face-Mask-Detection
### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Result](#result)
5. [How to run](#run)

## Introduction<a name="introduction"></a>
In the COVID-19 pandemic, wearing the mask is an effective and economical measure to prevent the spread of the virus, which led to the special concern for the face masked detection ability of the machine. Regardless of the topicality of that task, face detection with various occlusion degrees has been interested for several years, when older and smaller benchmark datasets for face detection like [FDBD](http://vis-www.cs.umass.edu/fddb/index.html) or [AFW](https://paperswithcode.com/dataset/afw) were addressed mostly by the state-of-the-art models. The contrast of the two tasks is that the first one ( or face masked detection) focus on detecting the bounding box of the faces and attempt to classify whether those faces are wearing the mask or not, or even wearing in the incorrect way, while the second one ( or masked face detection) just try to detect which region in the image contains a face, ignore the type of occlusion. As we can see, the second task creates a base stage for the first one and also deals with various facial problems as face recognition, emotion recognition, face alignment, etc. For that reason, the scope of our project is that just perform experiments that inference recent the state-of-the-art models like [MTCNN](https://github.com/ipazc/mtcnn), [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), [RetinaFace](https://github.com/serengil/retinaface), and [Yolov5Face](https://github.com/deepcam-cn/yolov5-face) on the face mask dataset.


## Dataset<a name="dataset"></a>
In this project, all models are evaluated on both testing set of [MAFA](https://openaccess.thecvf.com/content_cvpr_2017/html/Ge_Detecting_Masked_Faces_CVPR_2017_paper.html) and [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset).
  * [MAFA](https://openaccess.thecvf.com/content_cvpr_2017/html/Ge_Detecting_Masked_Faces_CVPR_2017_paper.html) contains 25,876 images in training set and 4,935 images in testing set. The testing set has 10,033 labels correspond to 6,354 masked, 996 unmask and 2,683 invalid face.
  * [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset) contains 7,385 images in traning set and 1,820 images in testing set. The testing set has 1,830 labels correspond to 993 masked, 791 unmasked and 46 invalid masked.
## Method<a name="method"></a>
As presented above, our experiments are executed on pretrained face detection model of MTCNN, SCRFD, RetinaFace and Yolov5Face. All of them were trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset which contains 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. The performance of models are judged using mAP (threshold IoU = 0.5) which was defined in [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). 
## Result<a name="result"></a>
<p float="left">
  <img src="/MAFA_AP.png" width="400" />
  <img src="/PWMFD_AP.png" width="400" /> 
</p>
Although our models do not classify what kind of detected face, we also count the detected objects followed by the original groundtruth to figure out performance of these models on each type of facial occlusion.

Result on testing set of MAFA:
| Model      | Masked  | Unmasked |Invalid face|
| ---------- |:-------:| :--------:| :----------:|
| MTCNN      |  4,497  |    816   |     733    |
| SCRFD      |  5,735  |    863   |   1,202    |
| RetinaFace |  5,895  |    861   |   1,279    |
| Yolov5Face |   5,971 |    877   |   1,334    |


Result on testing set of PWMFD:
| Model      | Masked  | Unmasked |Incorrect masked|
| ---------- |:-------:| :--------:| :----------:|
| MTCNN      |   877   |   770    |    41      |
| SCRFD      |   928   |   641    |    46      |
| RetinaFace |   877   |   774    |    41      |
| Yolov5Face |   981   |   787    |    46      |

## How to run<a name="run"></a>
### Install requirements 
* All requirement packages are included in requirement.txt
* To clone the repository and install requirement packages, run the script below
```bash
git clone https://github.com/ldvkhanh2001/Experiment-Face-Mask-Detection
cd Experiment-Face-Mask-Detection
pip install requirement.txt
```
### Evaluation
1. Download testing set of [MAFA](https://www.kaggle.com/rahulmangalampalli/mafa-data) and [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset), then store them in path `/data/<dataset>/images`
2. To standardize the annotations of those dataset, run
```
python utils/data_annotations.py
```
3. To inference MTCNN, SCRFD and RetinaFace, run
```
python evaluate/<model>.py
```
4. To inference Yolov5Face, run
```
python evaluate/YOLOv5Face/yolov5-face/Yolov5Face.py
```
5. Finally, run `main.py` to evaluate all models on both dataset.




