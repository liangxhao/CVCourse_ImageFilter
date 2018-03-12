from MainUI import MainUI
from PyQt5.QtWidgets import QFileDialog,QMessageBox
import os

from DataManager import DataManager
from SliderManager import SliderManager

class MainProgram(MainUI):
    def __init__(self,parent=None):
        super(MainProgram,self).__init__()
        self.__initVariable()



    def __initVariable(self):
        self.imageFile=None#带路径的名片名称
        self.imageManager=DataManager()

        self.sliderManager=SliderManager()
        self.sliderManager.setImageManager(self.imageManager)
        self.sliderManager.setViewer(self.centerUi)
        self.sliderManager.setScaleTextViewer(self.scaleText)
        self.sliderManager.setRateTextViewer(self.rateText)

        self.slider.valueChanged.connect(self.sliderManager.scaleChange)
        
        


    #打开
    def openImage(self):
        self.imageFile, ext = QFileDialog.getOpenFileName(self,  
                                    "文件选择",  
                                    '',  
                                    "Images (*.jpg *.bmp *.png)")          
        if len(self.imageFile)>0:
            title=self.title+'——'+os.path.basename(self.imageFile)
            self.setWindowTitle(title)
            self.imageManager.readImage(self.imageFile)
            image=self.imageManager.getOriginalImage()
            self.centerUi.setOriginalImage(image)
        else:
            self.imageFile=None


    #保存
    def saveImage(self):
        if not self.imageManager.canUse:
            return

        saveImageName, ext = QFileDialog.getSaveFileName(self,
                                                            "保存图片",
                                                            '',
                                                            "Image File (*.jpg)")
        if len(saveImageName) > 0:
            self.imageManager.saveImage(saveImageName)



    #链接图像的显示
    def linkImage(self):
        self.centerUi.changeLinkState()


    def enhance(self,idx):
        self.imageManager.setAlgorithms(idx)
        image = self.imageManager.getEnhance()
        self.centerUi.setEnhanceImage(image, 0)
        rate = self.imageManager.getReverseRate()
        self.rateText.setText(rate)





    def enhance1(self):
        if self.imageFile is None:
            return
        self.enhance(0)

        title = self.title+'——'+os.path.basename(self.imageFile)+'——guided filter'
        self.setWindowTitle(title)
        

    
    def enhance2(self):
        if self.imageFile is None:
            return

        self.enhance(1)

        title = self.title+'——'+os.path.basename(self.imageFile) + '——L0 smooth'
        self.setWindowTitle(title)


    def enhance3(self):
        if self.imageFile is None:
            return

        self.enhance(2)

        title = self.title + '——' + os.path.basename(self.imageFile) + '——Bilateral filter'
        self.setWindowTitle(title)