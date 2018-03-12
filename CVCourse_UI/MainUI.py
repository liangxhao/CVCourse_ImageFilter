from PyQt5.QtWidgets import QMainWindow,QAction,qApp,QSlider,QSpacerItem,QSizePolicy,QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from abc import ABCMeta, abstractmethod
from CenterUI import CenterUI
import sys,os

import imageTool

class MainUI(QMainWindow):
    __metaclass__ = ABCMeta

    def __init__(self,parent=None):
        super(MainUI,self).__init__()
        self.__initVariable()
        self.__initUI()
        


    def __initVariable(self):
        #获取工作目录
        if getattr(sys, 'frozen', False):
            self.programPath = os.path.dirname(sys.executable)
        elif __file__:
            self.programPath = os.path.dirname(__file__)   
        self.programPath=self.programPath.replace("\\",'/')#主程序所在目录

        self.title='计算机视觉课程'


    def __initUI(self):
        self.__createMenu()
        self.centerUi=CenterUI()          
        self.setCentralWidget(self.centerUi)
        self.showMaximized()      
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(':image/program.png'))

        

    def __createMenu(self):
        ###############################################文件
        openImageAction=QAction(QIcon(':image/openImage.png'),'&打开图片',self)
        openImageAction.triggered.connect(self.openImage)
        saveImageAction = QAction(QIcon(':image/saveImage.png'), '&保存图片', self)
        saveImageAction.triggered.connect(self.saveImage)
        linkImageAction=QAction(QIcon(':image/linkImage.png'),'&链接显示',self)
        linkImageAction.triggered.connect(self.linkImage)
        exitAction=QAction('&退出 ',self)
        exitAction.triggered.connect(qApp.quit)

        ##############################################增强
        enhance1Action=QAction(QIcon(':image/enhance1.png'),'&guided filter',self)
        enhance1Action.triggered.connect(self.enhance1)
        enhance2Action=QAction(QIcon(':image/enhance2.png'),'&L0 smooth',self)
        enhance2Action.triggered.connect(self.enhance2)
        enhance3Action = QAction(QIcon(':image/enhance3.png'), '&Bilateral filter', self)
        enhance3Action.triggered.connect(self.enhance3)

        ######################################################工具栏部件
        scaleLabel = QLabel(self)
        scaleLabel.setText('增强倍数：')
        scaleLabel.setAlignment(Qt.AlignCenter)

        self.scaleText=QLabel(self)
        self.scaleText.setText('3.0')
        self.scaleText.setAlignment(Qt.AlignCenter)

        rateLabel=QLabel(self)
        rateLabel.setText('梯度反转比例：')
        rateLabel.setAlignment(Qt.AlignCenter)

        self.rateText=QLabel(self)
        self.rateText.setText('--')
        self.rateText.setAlignment(Qt.AlignCenter)


        self.slider=QSlider(Qt.Horizontal, self)
        self.slider.setFixedWidth(300)
        self.slider.setMinimum(10)
        self.slider.setMaximum(100)
        self.slider.setValue(30)

        self.setStyleSheet(
            '''
                QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;           
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 13px;
                margin-top: -2px;
                margin-bottom: -2px;
                border-radius: 4px;
            }
            QSlider{
                margin-left: 10px;
            }
            
            
            '''
        )

        style1='''   
                border: 1px ;       
                margin-left: 18px;
                font-family: Microsoft YaHei;
                font-size: 14px;      
            '''
        scaleLabel.setStyleSheet(style1)
        rateLabel.setStyleSheet(style1)

        style2 = '''   
                    border: 1px ;       
                    margin-left: 5px;
                    font-family: Microsoft YaHei;
                    font-size: 14px;      
                '''

        self.scaleText.setStyleSheet(style2)
        self.rateText.setStyleSheet(style2)

        #菜单栏
        menubar=self.menuBar()

        #“文件”菜单
        fileMenu=menubar.addMenu('  &文件  ')
        fileMenu.addAction(openImageAction)
        fileMenu.addAction(saveImageAction)
        fileMenu.addAction(linkImageAction)
        fileMenu.addAction(exitAction)

        enhanceMenu=menubar.addMenu('  &增强  ')
        enhanceMenu.addAction(enhance1Action)
        enhanceMenu.addAction(enhance2Action)
        enhanceMenu.addAction(enhance3Action)

        space=QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding)
        #工具栏
        toolbar=self.addToolBar('工具')
        toolbar.addAction(openImageAction)
        toolbar.addAction(saveImageAction)
        toolbar.addAction(linkImageAction)
        toolbar.addAction(enhance1Action)
        toolbar.addAction(enhance2Action)
        toolbar.addAction(enhance3Action)

        toolbar.addWidget(self.slider)
        toolbar.addWidget(scaleLabel)
        toolbar.addWidget(self.scaleText)
        toolbar.addWidget(rateLabel)
        toolbar.addWidget(self.rateText)


    #打开
    @abstractmethod
    def openImage(self):pass

    # 保存
    @abstractmethod
    def saveImage(self):pass

    #链接图像的显示
    @abstractmethod
    def linkImage(self):pass


    @abstractmethod
    def enhance1(self):pass

    @abstractmethod
    def enhance2(self):pass

    @abstractmethod
    def enhance3(self):
        pass


  