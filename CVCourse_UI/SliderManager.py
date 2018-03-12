from DataManager import DataManager
class SliderManager(object):
    def __init__(self):
        pass

    def setViewer(self,viewer):
        self.imageViewer=viewer


    def setImageManager(self,manager):
        self.imageManager=manager

    def setScaleTextViewer(self,textViewer):
        self.scaleViewer=textViewer

    def setRateTextViewer(self, textViewer):
        self.rateViewer=textViewer




    def scaleChange(self,scale):
        scale=scale/10
        self.imageManager.setScale(scale)
        scale = str(round(scale, 1))
        self.scaleViewer.setText(scale)

        if not self.imageManager.canUse:
            return


        image=self.imageManager.getEnhance()
        self.imageViewer.setEnhanceImage(image,1)
        rate=self.imageManager.getReverseRate()
        self.rateViewer.setText(rate)



