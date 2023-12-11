from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import QRectF, Qt, pyqtSignal, QSize, QRect
from PyQt5.QtGui import QPainter, QPen, QFont, QColor
import sys

class CircularProgressBar(QWidget):
    # Create a signal to emit when the value changes
    valueChanged = pyqtSignal(int)

    def __init__(self, time_required,parent=None):
        super().__init__(parent)
        self.value = 0
        self.setMaximumWidth(200)
        self.setMaximumHeight(200)
        self.maxValue = time_required
    
    def sizeHint(self):
        return QSize(200, 200)  # Provide a default size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Use the widget's size to determine the drawing rect
        size = min(self.width(), self.height()) - 20  # Adjust for padding
        lineWidth = 15
        rect = QRect(10, 10, size, size)  # QRect is used instead of QRectF

        # Angles in drawArc are 1/16th of a degree, so we multiply by 16.
        # A full circle is 360 degrees, which is 5760 in 1/16th degrees.
        full_circle = 360 * 16

        # Calculate span_angle based on the current value
        # It linearly maps the value range [1, 3000] to the angle range [0, full_circle]
        span_angle = int((self.value / self.maxValue) * full_circle)
        start_angle = 90*16

        # Draw the background circle
        background_pen = QPen(Qt.gray, lineWidth)
        background_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(background_pen)
        painter.drawEllipse(rect)

        # Draw the progress arc
        progress_pen = QPen(Qt.green, lineWidth)
        progress_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(progress_pen)
        painter.drawArc(rect, start_angle, -span_angle)

        # Calculate the percentage of the ellipse filled
        percentage_filled = (self.value / self.maxValue) * 100

        # Draw the percentage text
        font = painter.font()
        font.setPointSize(10)  # Adjust size as needed
        painter.setFont(font)
        painter.setPen(Qt.black)
        painter.drawText(rect, Qt.AlignCenter, f"{percentage_filled:.0f}%")
        # print(f"value {self.value}, max_value {self.maxValue}, percentage={percentage_filled:.0f}%")

    def setValue(self, value):
        # Ensure value is between 0 and 3000
        self.value = max(0, min(self.maxValue, value))
        self.update()  # Trigger a repaint