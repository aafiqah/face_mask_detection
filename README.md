# Face Mask Detection and Alert Message System

This project implements a Face Mask Detection system with an alert messaging feature. 

## Requirements

To run this project, you'll need the following:

- [PyCharm IDE](https://www.jetbrains.com/pycharm/): An integrated development environment for Python.
- [Python 3](https://www.python.org/): The programming language used for the implementation.
- [Qt Designer](https://doc.qt.io/qt-5/qtdesigner-manual.html): A tool for designing graphical user interfaces.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
2. Open in PyCharm:
    - Open Pycharm IDE.
    - Choose "File" -> "Open Project" and select the cloned repository.

3. Install dependencies in terminal:
    ```bash
    pip install -r requirements.txt
4. Run the First Application:
    ```bash
    python train_mask_detector.py
5. Then, Run the Second Application:
    ```bash
    python FaceMaskDetector.py
## Features

- Face Mask Detection: The system uses computer vision to detect whether individuals in an image or video stream are wearing face masks.
- Alert Messaging: When a person is detected without a face mask, an alert message is triggered.
