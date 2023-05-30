import sys
import cv2
import numpy as np
from time import strftime
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.utils import img_to_array


def detect_and_predict_mask(image, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations, and
    # the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # opencv_python - 4.7.0.72

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all* faces at
        # the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("FaceMaskDetector.ui", self)
        self.datetime.setText(strftime("%H:%M:%S %d/%m/%Y"))
        self.browse_btn.clicked.connect(self.file_display)
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.control_btn.clicked.connect(self.controlTimer)

    def file_display(self):
        filepath = QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/user/Pictures')
        self.file_path.setText(filepath[0])
        self.cap = cv2.VideoCapture(filepath[0])
        ret, image2 = self.cap.read()
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        prototxtPath2 = r"face_detector/deploy.prototxt"
        weightsPath2 = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet2 = cv2.dnn.readNet(prototxtPath2, weightsPath2)
        maskNet2 = load_model("mask_detector.model")
        (locs, preds) = detect_and_predict_mask(image2, faceNet2, maskNet2)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw the bounding box and text
            label2 = "Mask" if mask > withoutMask else "No Mask"
            # (r,g,b)
            color2 = (0, 255, 0) if label2 == "Mask" else (255, 0, 0)
            # include the probability in the label
            label2 = "{}: {:.2f}%".format(label2, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output frame
            cv2.putText(image2, label2, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color2, 2)
            cv2.rectangle(image2, (startX, startY), (endX, endY), color2, 2)
            if mask > withoutMask:
                self.alert_msg.setText("WELL DONE")
            else:
                self.alert_msg.setText("PLEASE WEAR YOUR FACE MASK!!")

        height2, width2, channel2 = image2.shape
        step2 = channel2 * width2
        qImg2 = QImage(image2.data, width2, height2, step2, QImage.Format_RGB888)
        self.videoframe.setPixmap(QPixmap.fromImage(qImg2))

    def viewCam(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prototxtPath = r"face_detector/deploy.prototxt"
        weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        maskNet = load_model("mask_detector.model")
        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            # (r,g,b)
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            if mask > withoutMask:
                self.alert_msg.setText("WELL DONE")
            else:
                self.alert_msg.setText("PLEASE WEAR YOUR FACE MASK!!")

        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.videoframe.setPixmap(QPixmap.fromImage(qImg))

    def controlTimer(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.file_path.setText("")
            self.alert_msg.setText("")
            self.timer.start(20)
            self.control_btn.setText("CLOSE CAMERA")
        else:
            self.timer.stop()
            self.cap.release()
            self.control_btn.setText("OPEN CAMERA")


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
