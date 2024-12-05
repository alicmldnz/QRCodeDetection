**QR-Code Detection:** 

**Methods:** 

PyZbar: https://pypi.org/project/pyzbar/

OpenCv : https://docs.opencv.org/4.x/de/dc3/classcv\_1\_1QRCodeDetector.html 

Pyboof [:https://pypi.org/project/PyBoof/ ](https://pypi.org/project/PyBoof/)https://boofcv.org/index.php?title=Example\_Detect\_QR\_Code

Yolo: Pretrained Model was used but the dataset of the pre-trained model is unknown. 

1. **Demo** 

Methods were tested on PC according to the Dataset. 

**Accuracy Metric**: IoU 

**Dataset:** 1510 images (h[ttps://universe.roboflow.com/qr-lmsul/qr-code-detection - jz2e3/dataset/2) ](https://universe.roboflow.com/qr-lmsul/qr-code-detection-jz2e3/dataset/2)

![](results/dataset.png)

**Comprasion:** 

![](results/results.png)

**Example Results:** 

Bounding boxes were rearranged with extreme values to be comparable with labels. (minimum and maximum x and y coordinates) 

**PyZbar:** 

![](pyzbar/results/1.jpg)![](pyzbar/results/2.jpg)![](pyzbar/results/3.jpg)![](pyzbar/results/4.jpg)![](pyzbar/results/5.jpg)![](pyzbar/results/6.jpg)![](pyzbar/results/7.jpg)![](pyzbar/results/8.jpg)![](pyzbar/results/9.jpg)

In the 3rd example, it actually found 3 QR codes, but because of the way I chose to draw the bounding box, it seems like it found just one huge QR code.

**OpenCv :** 

![](opencv/results/1.jpg)![](opencv/results/2.jpg)![](opencv/results/3.jpg)![](opencv/results/4.jpg)![](opencv/results/5.jpg)![](opencv/results/6.jpg)![](opencv/results/7.jpg)![](opencv/results/8.jpg)![](opencv/results/9.jpg)

**Pyboof:** 

![](pyboof/examples/results/1.jpg)![](pyboof/examples/results/2.jpg)![](pyboof/examples/results/3.jpg)![](pyboof/examples/results/4.jpg)![](pyboof/examples/results/5.jpg)![](pyboof/examples/results/6.jpg)![](pyboof/examples/results/7.jpg)![](pyboof/examples/results/8.jpg)![](pyboof/examples/results/9.jpg)!

In the 3rd example it found 4 qr codes.

**Yolo:** 

![](yolo/results/1.jpg)![](yolo/results/2.jpg)![](yolo/results/3.jpg)![](yolo/results/4.jpg)![](yolo/results/5.jpg)![](yolo/results/6.jpg)![](yolo/results/7.jpg)![](yolo/results/8.jpg)![](yolo/results/9.jpg)!

2. **Demo** 

I didn't have a dataset available, so I used a computer camera for real-time captures instead.

**OpenCV:** It is not successful in images shown far from the camera. It is less successful in images shown at an angle to the camera. It detects rotated images successfully.

**PyZbar:** It is not successful in images shown from far away. It is successful in images shown at an angle to the camera. It is successful up close. It detects rotated images successfully. 

**Yolo:** When I hold the picture very close, it can be found. Otherwise, it cannot be found. It often cannot distinguish two pictures side by side. Rotated pictures were 

successfully found. It is also successfull when shown at an angle. (Close-ups) 

**Pyboof:** Image (grayscale) fps is quite low. It can be said that it is successful in close - up and successful in small pictures (from a distance). Rotated is successful.

**Extra Comprasion:** 

[https://www.dynamsoft.com/codepool/qr-code-reading - benchmark-and-comparison.html ]

![](results/comprasion.png)


**Some Resources:** 

**Pyboof example:** <https://github.com/lessthanoptimal/PyBoof/tree/SNAPSHOT>

**Improvement (maybe for future ) : [https://github.com/Asadullah - Dal17/Improved_detection-with-Optical-Flow-distance-estimation ](https://github.com/Asadullah-Dal17/Improved_detection-with-Optical-Flow-distance-estimation)**

**Detection using Yolo:** [https://github.com/ErenKaymakci/Real-Time-QR-Detection - and-Decoding/blob/main/README.md ](https://github.com/ErenKaymakci/Real-Time-QR-Detection-and-Decoding/blob/main/README.md)

[https://github.com/yushulx/opencv-yolo-qr-detection ](https://github.com/yushulx/opencv-yolo-qr-detection)

**Different Dataset can be used:** https://www.kaggle.com/datasets/hamidl/yoloqrlabeled/data
