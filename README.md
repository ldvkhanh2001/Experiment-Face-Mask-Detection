# Experiment-Face-Mask-Detection
### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Result](#result)
5. [How to run](#run)

## Introduction
In the COVID-19 pandemic, wearing mask is an effective and economical measure to prevent the spread of virus, that led to the special concern for face masked detection ability of machine. Regardless the topicality of that task, face detection with various of occlusion degree has been interested for several years, when older and smaller benchmark datasets for face detection like [FDBD](http://vis-www.cs.umass.edu/fddb/index.html) or [AFW](https://paperswithcode.com/dataset/afw) were addressed atmostly by the-state-of-the-art models. The contrast of two task is that the first one ( or face masked detection) focus on detecting bounding box of the faces and attempt to classify whether those faces are wearing mask or not, or even wearing in incorrect way, while second one ( or masked face detection) just try to detect which region in the image contain face, ignore type of occlusion. As we can see, the second task creates a base stage for the first one, and also deals with various facial problems as face recognition, emotion recognition, face alignment, etc. By that reason, the scope of our project is that just perform experiments which inference recent the state-of-the-art models like MTCNN, SCRFD, RetinaFace and Yolov5Face  on face mask dataset.


## Dataset
In this project, all models are evaluated on both testing set of [MAFA](https://openaccess.thecvf.com/content_cvpr_2017/html/Ge_Detecting_Masked_Faces_CVPR_2017_paper.html) and [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset).
  * [MAFA](https://openaccess.thecvf.com/content_cvpr_2017/html/Ge_Detecting_Masked_Faces_CVPR_2017_paper.html) contains 25,876 images in training set and 4,935 images in testing set. The testing set has 10,033 labels correspond to 6354 masked, 996 unmask and 2683 invalid face.
  * [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset) contains 7385 images in traning set and 1820 images in testing set. The testing set has 1830 labels correspond to 993 masked, 791 unmasked and 46 invalid masked.
## Method
As presented above, our experiments are executed on pretrained face detection model of MTCNN, SCRFD, RetinaFace and Yolov5Face. All of them were trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset which contains 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. The performance of models are judged using mAP (threshold IoU = 0.5) which was defined in [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). 
## Result
<p float="left">
  <img src="/MAFA_AP.png" width="400" />
  <img src="/PWMFD_AP.png" width="400" /> 
</p>
Although our models do not classify what kind of detected face, we also count the detected objects followed by the original groundtruth to figure out performance of those models on each type of facial occlusion.

Result on testing set of MAFA:
| Model      | Masked  | Unmasked |Invalid face|
| ---------- |:-------:| --------:| ----------:|
| MTCNN      |   20    | $1600    |  20        |
| SCRFD      |    20   |   $12    |    20      |
| RetinaFace |    20   |    $1    |      20    |
| Yolov5Face |     20  |    $1    |        20  |


Result on testing set of PWMFD:
| Model      | Masked  | Unmasked |Incorrect masked|
| ---------- |:-------:| --------:| ----------:|
| MTCNN      |   20    | $1600    |  20        |
| SCRFD      |    20   |   $12    |    20      |
| RetinaFace |    20   |    $1    |      20    |
| Yolov5Face |     20  |    $1    |        20  |

## How to run



