import numpy as np 
import CVCourse
from PIL import Image,ImageQt
from PyQt5.QtGui import QImage
# import cv2

class DataManager(object):
    def __init__(self):
        self.__initVariable()
        

    def __initVariable(self):
        #原始图像
        self.__original=None
        #滤波图像
        # self.__filter=None
        #细节图像
        # self.__details=None

        self.__cv=CVCourse.Filter()

        self.__scale=3.0

        self.canUse=False

        self.__reverseRate=None
        #增强后的图
        self.__enhance=None
        #算法编号
        self.__idx=None

        # self.__alpha=None

    def setScale(self,scale):
        self.__scale=scale


    def setAlgorithms(self,idx):
        self.__idx=idx
    


    #读取图像
    def readImage(self,file):
        img=Image.open(file).convert("RGB")

        self.__original=np.array(img,dtype=np.uint8)
        self.__cv.initData(self.__original)




    def getOriginalImage(self):
        return self.__numpyToQimage(self.__original)


    
    def getEnhance(self):
        self.__enhance=self.__cv.filter(self.__idx,self.__scale)
        # cv2.imshow('sss',self.__enhance)
        # cv2.waitKey(0)
        self.canUse=True

        return self.__numpyToQimage(self.__enhance)


    def getReverseRate(self):
        value=self.__cv.getRate()*100

        text=str(round(value,2))+'%'

        return text



    def __numpyToQimage(self,array):
        if array.ndim==2:
            image=QImage(array,array.shape[1],array.shape[0],QImage.Format_Indexed8)
        else:

            image = Image.fromarray(array)
            image = ImageQt.ImageQt(image).convertToFormat(QImage.Format_RGB888)
        return image




    def saveImage(self,file):
        image=Image.fromarray(self.__enhance)
        image.save(file)




