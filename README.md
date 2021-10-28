# Experiment-Face-Mask-Detection
### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Result](#result)
5. [How to run](#run)

## Introduction
In the COVID-19 pandemic, wearing mask is an effective and economical measure to prevent the spread of virus, that led to the special concern for face masked detection ability of machine. Regardless the topicality of that task, face detection with various of occlusion degree has been interested for several years, when older and smaller benchmark datasets for face detection like [FDBD](http://vis-www.cs.umass.edu/fddb/index.html) or [AFW](https://paperswithcode.com/dataset/afw) were addressed atmostly by the-state-of-the-art models. The contrast of two task is that the first one ( or face masked detection) focus on detecting bounding box of the faces and attempt to classify whether those faces are wearing mask or not, or even wearing in incorrect way, while second one ( or masked face detection) just try to detect which region in the image contain face, ignore type of occlusion. As we can see, the second task creates a base stage for the first one, and also deals with various facial problems as face recognition, emotion recognition, face alignment, etc. By that reason, the scope of our project is that just perform experiments which inference recent the state-of-the-art models like MTCNN, SCRFD, RetinaFace and Yolov5Face  on face mask dataset.

<p align="center">
  <img src="https://drive.google.com/file/d/10mGjuzNNXH6cWgNAxH1u73OhkSJQW9de/view?usp=sharing" width="450" height="300" />
</p>
