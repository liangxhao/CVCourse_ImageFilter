from ImageView import ImageView
from PyQt5.QtWidgets import QHBoxLayout, QWidget,QSplitter,QDesktopWidget,QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PIL import ImageQt,Image
import os


class CenterUI(QWidget):
    def __init__(self):
        super().__init__()
        self.__initVariable()
        self.__initUI()


    def __initVariable(self):
        self.linkView=True


    def __initUI(self):
        self.viewerOriginal=ImageView(self)
        self.viewerEnhance=ImageView(self)

        self.connect_signal_slot()

        #控件宽度
        width=self.width()
        height=self.height()
        ratioMain=[0.5,0.5]
        sizeMain=[width*ratioMain[0],width*ratioMain[1]] 
        

        splitter=QSplitter(Qt.Horizontal,self)
        splitter.addWidget(self.viewerOriginal)  
        splitter.addWidget(self.viewerEnhance)  

        splitter.setSizes(sizeMain)    

        hbox=QHBoxLayout()
        hbox.addWidget(splitter)
        
        self.setLayout(hbox)


    #连接信号与槽
    def connect_signal_slot(self):
        #1连接2
        self.viewerOriginal.wheelZoom.connect(self.viewerEnhance.accept_wheelZoom)
        self.viewerOriginal.mouseButtonPressed.connect(self.viewerEnhance.accept_mousePress)
        self.viewerOriginal.mouseButtonMove.connect(self.viewerEnhance.accept_mouseMove)
        self.viewerOriginal.mouseButtonRelease.connect(self.viewerEnhance.accept_mouseRelease)

        #2连接1
        self.viewerEnhance.wheelZoom.connect(self.viewerOriginal.accept_wheelZoom)
        self.viewerEnhance.mouseButtonPressed.connect(self.viewerOriginal.accept_mousePress)
        self.viewerEnhance.mouseButtonMove.connect(self.viewerOriginal.accept_mouseMove)
        self.viewerEnhance.mouseButtonRelease.connect(self.viewerOriginal.accept_mouseRelease)


    #关闭信号与槽的连接
    def disconnect_signal_slot(self):
        #1连接2
        self.viewerOriginal.wheelZoom.disconnect(self.viewerEnhance.accept_wheelZoom)
        self.viewerOriginal.mouseButtonPressed.disconnect(self.viewerEnhance.accept_mousePress)
        self.viewerOriginal.mouseButtonMove.disconnect(self.viewerEnhance.accept_mouseMove)
        self.viewerOriginal.mouseButtonRelease.disconnect(self.viewerEnhance.accept_mouseRelease)

        #2连接1
        self.viewerEnhance.wheelZoom.disconnect(self.viewerOriginal.accept_wheelZoom)
        self.viewerEnhance.mouseButtonPressed.disconnect(self.viewerOriginal.accept_mousePress)
        self.viewerEnhance.mouseButtonMove.disconnect(self.viewerOriginal.accept_mouseMove)
        self.viewerEnhance.mouseButtonRelease.disconnect(self.viewerOriginal.accept_mouseRelease)



    
    def changeLinkState(self):
        if self.linkView:
            self.linkView=False
            self.disconnect_signal_slot()
        else:
            self.linkView=True
            self.connect_signal_slot()
            self.viewerOriginal.clearMoveAndZoom()
            self.viewerEnhance.clearMoveAndZoom()



    def setOriginalImage(self,file):

        self.viewerOriginal.clearMoveAndZoom()
        self.viewerEnhance.clearMoveAndZoom()
        self.viewerOriginal.clearImage()
        self.viewerEnhance.clearImage()

        if isinstance(file,str):
            if not os.path.exists(file):
                QMessageBox.information(self,'失败','图像不存在，请检查路径')
                return

            frame=Image.open(file)
            img=ImageQt.ImageQt(frame).convertToFormat(QImage.Format_RGB888)
            self.viewerOriginal.setImage(img)

        else:
            self.viewerOriginal.setImage(file)


    def setEnhanceImage(self,file,mode):
        #第一次读取图像
        if mode==0:
            if self.linkView:
                self.viewerOriginal.clearMoveAndZoom()
            self.viewerEnhance.clearMoveAndZoom()
            self.viewerEnhance.clearImage()
        else:#更新增项图像
            # self.viewerEnhance.clearImage()
            pass

        if isinstance(file,str):
            if not os.path.exists(file):
                QMessageBox.information(self,'失败','图像不存在，请检查路径')
                return

            frame=Image.open(file)
            img=ImageQt.ImageQt(frame).convertToFormat(QImage.Format_RGB888)
            self.viewerEnhance.setImage(img)

        else:
            self.viewerEnhance.setImage(file)


        


