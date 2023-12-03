import sys
import cv2
from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QImage, QPainter
import movenet

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method

        uic.loadUi('ui/MainWindow.ui', self) # Load the .ui file

        # Add CameraWidget to the main window
        self.camera_widget = CameraWidget()
        self.cameraWidget.setLayout(QtWidgets.QVBoxLayout())
        self.cameraWidget.layout().addWidget(self.camera_widget)


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Create a timer for updating the feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

        self.image = None

    def initUI(self):
        self.resize(640, 480)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = movenet.process_image(frame)

            self.image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.update()  # Trigger paint event

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            painter.drawImage(0, 0, self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.cap.release()


app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
window.show()
app.exec_() # Start the application