from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import QRectF, Qt, pyqtSignal, QSize, QRect
from PyQt5.QtGui import QPainter, QPen, QFont, QColor
import sys

class CircularProgressBar(QWidget):
    # Create a signal to emit when the value changes
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setMaximumWidth(200)
        self.setMaximumHeight(200)
    
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
        start_angle = 90 * 16  # Start angle is 90 degrees above the horizontal
        span_angle = int(-360 * 16 * self.value / 100)  # Span angle should be an integer

        # Draw the background circle
        background_pen = QPen(Qt.gray, lineWidth)
        background_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(background_pen)
        painter.drawEllipse(rect)

        # Draw the progress arc
        progress_pen = QPen(Qt.green, lineWidth)
        progress_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(progress_pen)
        painter.drawArc(rect, start_angle, span_angle)

        # Draw the percentage text
        font = painter.font()
        font.setPointSize(10)  # Adjust size as needed
        painter.setFont(font)
        painter.setPen(Qt.black)
        painter.drawText(rect, Qt.AlignCenter, f"{self.value}%")

    def setValue(self, value):
        # Ensure value is between 0 and 100
        self.value = max(0, min(100, value))
        self.update()  # Trigger a repaint