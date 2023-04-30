# Object Detection

Object Detection using Deep Learning with Convolutional Neural Network Model

https://user-images.githubusercontent.com/60769071/235369537-ac80c4d7-6c8c-49f5-b088-95386e3ac70c.mp4

# Repository Files
```
├── model
│   └── object_detector.h5
├── README.md
├── main.py
├── object_detection.ipynb
├── object_detector.py
└── utils.py
```

# Project Setup
```
Python 3.9.16 
   
Package Requirements
   -  tensorflow
   -  tensorflow-datasets
   -  opencv-python
   -  numpy
```

# Dataset
The dataset used in this project is the <a href = "https://www.tensorflow.org/datasets/catalog/voc">voc</a> dataset from TensorFlow Datasets.

# Run
Please run main.py to start a basic webapp for project demo.
```
streamlit run main.py
```
<b> Note: </b> The current working directory must be where the files are located and the model <b>must be</b> trained first.

To train the mode, please check out the jupyter notebook.
```
object_detection.ipynb
```
<b> Note: </b> The trained model can be found in the directory `model`.
