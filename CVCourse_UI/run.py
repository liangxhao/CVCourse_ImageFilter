import sys
from PyQt5.QtWidgets import QApplication
from MainProgram import MainProgram


if __name__=='__main__':

    app = QApplication(sys.argv)
    program=MainProgram()
    program.show()
    
    
    sys.exit(app.exec_())