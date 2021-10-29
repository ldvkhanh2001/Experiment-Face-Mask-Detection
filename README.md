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

