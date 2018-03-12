from PyQt5.QtWidgets import QGraphicsView,QGraphicsScene

from PyQt5.QtCore import Qt, QRectF,pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage, QPixmap,QTransform,QMouseEvent


class ImageView(QGraphicsView):
    wheelZoom=pyqtSignal(float)#滚轮滚动

    mouseButtonPressed=pyqtSignal(QMouseEvent)#鼠标按压
    mouseButtonMove=pyqtSignal(QMouseEvent)#鼠标按压
    mouseButtonRelease=pyqtSignal(QMouseEvent)#鼠标按压


    def __init__(self,parent=None):
        super(ImageView,self).__init__()

        self.scene=QGraphicsScene()
        self.setScene(self.scene)
        self._pixmapHandle = None
        self.aspectRatioMode = Qt.KeepAspectRatio

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self.zoom=-1
       
      

    def hasImage(self):
        return self._pixmapHandle is not None


    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None
            self.zoom=-1
            self.scene.clear()



    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None



    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
            """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()
    


    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if self.zoom<0:
            if self.size().width()>self.size().height():
                self.zoom = self.size().height() / self.scene.height()
            else:
                self.zoom = self.size().width() / self.scene.width()

        self.setTransform(QTransform().scale(self.zoom, self.zoom))

   

    def wheelEvent(self,event):
        moose=event.angleDelta().y()

        if moose>0:
            self.zoomIn()
        elif moose<0:
            self.zoomOut()

        if self.hasImage():
            self.wheelZoom.emit(self.zoom)



    @pyqtSlot(float)
    def accept_wheelZoom(self,zoom):
        if self.hasImage():
            self.zoom=zoom
            self.updateViewer()


    def zoomIn(self):
        self.zoom *= 1.05
        self.updateViewer()


    def zoomOut(self):
        self.zoom /= 1.05
        self.updateViewer()   


    #鼠标按下
    def mousePressEvent(self, event):    
        if self.hasImage(): 
            self.mouseButtonPressed.emit(event)#发送鼠标左键按压信号

        QGraphicsView.mousePressEvent(self, event)
    


    @pyqtSlot(QMouseEvent)
    def accept_mousePress(self,event):
        if self.hasImage():
            QGraphicsView.mousePressEvent(self, event)



    #鼠标移动
    def mouseMoveEvent(self,event):
        if self.hasImage():
            self.mouseButtonMove.emit(event)#发送鼠标左键移动信号
        QGraphicsView.mouseMoveEvent(self,event)
    


    @pyqtSlot(QMouseEvent)
    def accept_mouseMove(self,event):
        if self.hasImage():
            QGraphicsView.mouseMoveEvent(self,event)



    #鼠标释放
    def mouseReleaseEvent(self, event):  
        if self.hasImage():
            self.mouseButtonRelease.emit(event)#发送鼠标左键释放信号
        QGraphicsView.mouseReleaseEvent(self, event)
             

    @pyqtSlot(QMouseEvent)    
    def accept_mouseRelease(self, event):
        if self.hasImage():
            QGraphicsView.mouseReleaseEvent(self, event)



    def clearMoveAndZoom(self):
        self.zoom=-1
        self.updateViewer()