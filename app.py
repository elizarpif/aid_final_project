import sys
import cv2
from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTime, QSize
import movenet
from circular_progress_bar import CircularProgressBar

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method

        uic.loadUi('ui/MainWindow.ui', self) # Load the .ui file

        self.progressBar.setValue(0)

        self.circularProgressBarUp = CircularProgressBar()
        self.circularProgressBarDown = CircularProgressBar()
        self.circularProgressBarPerp = CircularProgressBar()

        self.horizontalLayout.layout().addWidget(self.circularProgressBarDown)
        self.horizontalLayout.layout().addWidget(self.circularProgressBarPerp)
        self.horizontalLayout.layout().addWidget(self.circularProgressBarUp)

        # Add CameraWidget to the main window
        self.camera_widget = CameraWidget(self.circularProgressBarDown, self.circularProgressBarPerp, self.circularProgressBarUp, self.progressBar)
        self.cameraWidget.setLayout(QtWidgets.QVBoxLayout())
        self.cameraWidget.layout().addWidget(self.camera_widget)

         # Connect start and end buttons to the exercise methods
        self.startBtn.clicked.connect(self.camera_widget.startExercise)
        self.endBtn.clicked.connect(self.camera_widget.endExercise)

class CameraWidget(QWidget):
    def __init__(self, dialDown, dialPerp, dialUp, progressBar):
        super().__init__()

        self.initUI()


        self.pose_time_required = 3000  # 3 seconds in milliseconds


        # dials, progress bars
        self.dialDown = dialDown
        self.dialPerp = dialPerp
        self.dialUp = dialUp
        self.progressBar = progressBar

        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Create a timer for updating the feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

        self.image = None
        self.is_exercising = False
        self.initLeftHandExercise()

    def initUI(self):
        self.resize(640, 480)

    def initLeftHandExercise(self):
        self.pose_sequence = ["down", "perp", "up"]
        self.current_pose_index = 0
        self.pose_timer = QTime()

    def updateUI(self, pose):
        # Update progress bar
        progress = (self.current_pose_index + 1) * 33
        self.progressBar.setValue(min(progress, 100))

    def startExercise(self):
        self.is_exercising = True
        self.current_pose_index = 0
        self.pose_timer.start()
        self.progressBar.setValue(0)


    def endExercise(self):
        self.is_exercising = False

    def checkPose(self, processed_image):
        current_pose = self.pose_sequence[self.current_pose_index]
        if processed_image.is_pose(current_pose):
            elapsed_time = self.pose_timer.elapsed()
            self.updateDial(current_pose, elapsed_time)
            if elapsed_time >= self.pose_time_required:
                self.updateUI(current_pose)
                self.current_pose_index += 1
                if self.current_pose_index < len(self.pose_sequence):
                    self.pose_timer.restart()  # Restart timer for the next pose
                else:
                    self.endExercise()  # End exercise if all poses are done
        else:
            # Reset the dials if the pose is not correct
            self.resetDials()
            self.pose_timer.restart()

    def updateDial(self, pose, value):
        if pose == "down":
            self.dialDown.setValue(value)
        elif pose == "perp":
            self.dialPerp.setValue(value)
        elif pose == "up":
            self.dialUp.setValue(value)

    def resetDials(self):
        self.dialDown.setValue(0)
        self.dialPerp.setValue(0)
        self.dialUp.setValue(0)


    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.is_exercising:
                processed_image = movenet.process_image(frame)
                self.checkPose(processed_image)

                # frame = processedImage.img

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