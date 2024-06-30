from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import RandomForestClassifier
from ModeloML import modelo

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1900, 1800))
        MainWindow.setMaximumSize(QtCore.QSize(1900, 1800))
        MainWindow.setStyleSheet("background-color: rgb(82, 165, 247);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.frame_titulo = QtWidgets.QFrame(self.centralwidget)
        self.frame_titulo.setMinimumSize(QtCore.QSize(0, 180))
        self.frame_titulo.setMaximumSize(QtCore.QSize(16777215, 180))
        self.frame_titulo.setStyleSheet("background-color: rgb(72, 147, 217);")
        self.frame_titulo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_titulo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_titulo.setObjectName("frame_titulo")
        
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_titulo)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.frame_titulo_2 = QtWidgets.QFrame(self.frame_titulo)
        self.frame_titulo_2.setMaximumSize(QtCore.QSize(16777215, 180))
        self.frame_titulo_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_titulo_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_titulo_2.setObjectName("frame_titulo_2")
        
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_titulo_2)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        
        self.label_nombre = QtWidgets.QLabel(self.frame_titulo_2)
        self.label_nombre.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_nombre.setFont(font)
        self.label_nombre.setAlignment(QtCore.Qt.AlignCenter)
        self.label_nombre.setWordWrap(True)
        self.label_nombre.setObjectName("label_nombre")
        self.verticalLayout_6.addWidget(self.label_nombre)
        
        self.frame_2 = QtWidgets.QFrame(self.frame_titulo_2)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 60))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setContentsMargins(0, 0, 0, -1)
        self.gridLayout.setObjectName("gridLayout")
        
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setStyleSheet("background-color: rgb(205, 230, 255); border: 2px solid blue;")
        self.pushButton.setMinimumSize(QtCore.QSize(300, 75))
        self.pushButton.setMaximumSize(QtCore.QSize(300, 75))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        
        self.verticalLayout_6.addWidget(self.frame_2)
        self.horizontalLayout.addWidget(self.frame_titulo_2)
        self.verticalLayout.addWidget(self.frame_titulo)
        
        self.frame_cuerpo = QtWidgets.QFrame(self.centralwidget)
        self.frame_cuerpo.setMinimumSize(QtCore.QSize(0, 800))
        self.frame_cuerpo.setStyleSheet("background-color: rgb(72, 147, 217);")
        self.frame_cuerpo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_cuerpo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_cuerpo.setObjectName("frame_cuerpo")
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_cuerpo)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.frame_imgs = QtWidgets.QFrame(self.frame_cuerpo)
        self.frame_imgs.setMinimumSize(QtCore.QSize(0, 800))
        self.frame_imgs.setMaximumSize(QtCore.QSize(16777215, 800))
        # self.frame_imgs.setStyleSheet("background-color: rgb(205, 230, 255);")
        self.frame_imgs.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_imgs.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_imgs.setObjectName("frame_imgs")
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_imgs)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.frame = QtWidgets.QFrame(self.frame_imgs)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.frame_izq = QtWidgets.QFrame(self.frame)
        self.frame_izq.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_izq.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_izq.setObjectName("frame_izq")
        
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_izq)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        
        self.frame_imgor = QtWidgets.QFrame(self.frame_izq)
        self.frame_imgor.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_imgor.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_imgor.setObjectName("frame_imgor")
        
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_imgor)
        self.gridLayout_4.setObjectName("gridLayout_4")
        
        self.label_imgOR = MatplotlibWidget(self.frame_imgor)
        #self.label_imgOR.setObjectName("label_imgOR")
        
        self.gridLayout_4.addWidget(self.label_imgOR, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame_imgor)
        self.label_8.setMinimumSize(QtCore.QSize(0, 40))
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 0, 0, 1, 1)
        self.gridLayout_4.setContentsMargins(9, -1, 9, 9)
        self.gridLayout_4.setHorizontalSpacing(0)
        self.gridLayout_4.setVerticalSpacing(10)
        
        self.verticalLayout_3.addWidget(self.frame_imgor)
        self.horizontalLayout_3.addWidget(self.frame_izq)
        
        self.frame_der = QtWidgets.QFrame(self.frame)
        self.frame_der.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_der.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_der.setObjectName("frame_der")
        
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_der)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        
        self.frame_3 = QtWidgets.QFrame(self.frame_der)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_6.setContentsMargins(9, -1, 9, 9)
        self.gridLayout_6.setHorizontalSpacing(0)
        self.gridLayout_6.setVerticalSpacing(10)
        self.gridLayout_6.setObjectName("gridLayout_6")
        
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setMinimumSize(QtCore.QSize(0, 40))
        self.label.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_6.addWidget(self.label, 1, 0, 1, 1)

        self.label_imgSeg = MatplotlibWidget(self.frame_3)
        #self.label_imgSeg.setObjectName("label_imgSeg")

        self.gridLayout_6.addWidget(self.label_imgSeg, 2, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.frame_3)
        self.horizontalLayout_3.addWidget(self.frame_der)
        self.horizontalLayout_2.addWidget(self.frame)
        self.verticalLayout_2.addWidget(self.frame_imgs)
        
        # self.label_imgOR.setMinimumSize(QtCore.QSize(0, 500))
        # self.label_imgOR.setMaximumSize(QtCore.QSize(16777215, 700))

        # self.label_imgSeg.setMinimumSize(QtCore.QSize(0, 500))
        # self.label_imgSeg.setMaximumSize(QtCore.QSize(16777215, 700))
        
        self.frame_estadistica = QtWidgets.QFrame(self.frame_cuerpo)
        self.frame_estadistica.setMinimumSize(QtCore.QSize(0, 700))
        self.frame_estadistica.setMaximumSize(QtCore.QSize(16777215, 800))
        self.frame_estadistica.setStyleSheet("background-color: rgb(205, 230, 255);")
        self.frame_estadistica.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_estadistica.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_estadistica.setObjectName("frame_estadistica")
        
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_estadistica)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        
        self.label_2 = QtWidgets.QLabel(self.frame_estadistica)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 70))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_5.addWidget(self.label_2)
        
        self.label_7 = QtWidgets.QLabel(self.frame_estadistica)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 70))
        self.label_7.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_5.addWidget(self.label_7)
        
        self.frame_4 = QtWidgets.QFrame(self.frame_estadistica)
        self.frame_4.setMinimumSize(QtCore.QSize(0, 300))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        
        self.frame_5 = QtWidgets.QFrame(self.frame_4)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        
        self.label_irregularidad = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_irregularidad.setFont(font)
        self.label_irregularidad.setAlignment(QtCore.Qt.AlignCenter)
        self.label_irregularidad.setWordWrap(True)
        self.label_irregularidad.setObjectName("label_irregularidad")
        self.gridLayout_3.addWidget(self.label_irregularidad, 1, 0, 1, 1)
        
        self.label_ref_cant = QtWidgets.QLabel(self.frame_5)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # # font.setBold(False)
        # font.setWeight(75)
        # self.label_ref_cant.setFont(font)
        self.label_ref_cant.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_cant.setWordWrap(True)
        self.label_ref_cant.setObjectName("label_ref_cant")
        self.gridLayout_3.addWidget(self.label_ref_cant, 4, 2, 1, 1)
        
        self.label_ref_circu = QtWidgets.QLabel(self.frame_5)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_circu.setFont(font)
        self.label_ref_circu.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_circu.setWordWrap(True)
        self.label_ref_circu.setObjectName("label_ref_circu")
        self.gridLayout_3.addWidget(self.label_ref_circu, 2, 2, 1, 1)
        
        self.label_ref_asimetria = QtWidgets.QLabel(self.frame_5)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_asimetria.setFont(font)
        self.label_ref_asimetria.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_asimetria.setWordWrap(True)
        self.label_ref_asimetria.setObjectName("label_ref_asimetria")
        self.gridLayout_3.addWidget(self.label_ref_asimetria, 0, 2, 1, 1)
        
        self.label_circulo = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_circulo.setFont(font)
        self.label_circulo.setAlignment(QtCore.Qt.AlignCenter)
        self.label_circulo.setWordWrap(True)
        self.label_circulo.setObjectName("label_circulo")
        self.gridLayout_3.addWidget(self.label_circulo, 2, 0, 1, 1)
        
        self.label_valor_bordes = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_bordes.setFont(font)
        self.label_valor_bordes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_bordes.setWordWrap(True)
        self.label_valor_bordes.setObjectName("label_valor_bordes")
        self.gridLayout_3.addWidget(self.label_valor_bordes, 3, 1, 1, 1)
        
        self.label_cantcolores = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_cantcolores.setFont(font)
        self.label_cantcolores.setAlignment(QtCore.Qt.AlignCenter)
        self.label_cantcolores.setWordWrap(True)
        self.label_cantcolores.setObjectName("label_cantcolores")
        self.gridLayout_3.addWidget(self.label_cantcolores, 4, 0, 1, 1)
        
        self.label_valor_cant = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_cant.setFont(font)
        self.label_valor_cant.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_cant.setWordWrap(True)
        self.label_valor_cant.setObjectName("label_valor_cant")
        self.gridLayout_3.addWidget(self.label_valor_cant, 4, 1, 1, 1)
        
        self.label_bordes = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_bordes.setFont(font)
        self.label_bordes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bordes.setWordWrap(True)
        self.label_bordes.setObjectName("label_bordes")
        self.gridLayout_3.addWidget(self.label_bordes, 3, 0, 1, 1)
        
        self.label_ref_bordes = QtWidgets.QLabel(self.frame_5)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_bordes.setFont(font)
        self.label_ref_bordes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_bordes.setWordWrap(True)
        self.label_ref_bordes.setObjectName("label_ref_bordes")
        self.gridLayout_3.addWidget(self.label_ref_bordes, 3, 2, 1, 1)
        
        self.label_valor_irr = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_irr.setFont(font)
        self.label_valor_irr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_irr.setWordWrap(True)
        self.label_valor_irr.setObjectName("label_valor_irr")
        self.gridLayout_3.addWidget(self.label_valor_irr, 1, 1, 1, 1)
        
        self.label_ref_irr = QtWidgets.QLabel(self.frame_5)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_irr.setFont(font)
        self.label_ref_irr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_irr.setWordWrap(True)
        self.label_ref_irr.setObjectName("label_ref_irr")
        self.gridLayout_3.addWidget(self.label_ref_irr, 1, 2, 1, 1)
        
        self.label_valor_asimetria = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_asimetria.setFont(font)
        self.label_valor_asimetria.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_asimetria.setWordWrap(True)
        self.label_valor_asimetria.setObjectName("label_valor_asimetria")
        self.gridLayout_3.addWidget(self.label_valor_asimetria, 0, 1, 1, 1)
        
        self.label_valor_circ = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_circ.setFont(font)
        self.label_valor_circ.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_circ.setWordWrap(True)
        self.label_valor_circ.setObjectName("label_valor_circ")
        self.gridLayout_3.addWidget(self.label_valor_circ, 2, 1, 1, 1)
        
        self.label_asimetria = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_asimetria.setFont(font)
        self.label_asimetria.setAlignment(QtCore.Qt.AlignCenter)
        self.label_asimetria.setWordWrap(True)
        self.label_asimetria.setObjectName("label_asimetria")
        self.gridLayout_3.addWidget(self.label_asimetria, 0, 0, 1, 1)
        
        self.horizontalLayout_4.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.frame_4)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        
        self.label_ref_en = QtWidgets.QLabel(self.frame_6)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_en.setFont(font)
        self.label_ref_en.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_en.setWordWrap(True)
        self.label_ref_en.setObjectName("label_ref_en")
        self.gridLayout_5.addWidget(self.label_ref_en, 2, 2, 1, 1)
        
        self.label_entropia = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_entropia.setFont(font)
        self.label_entropia.setAlignment(QtCore.Qt.AlignCenter)
        self.label_entropia.setWordWrap(True)
        self.label_entropia.setObjectName("label_entropia")
        self.gridLayout_5.addWidget(self.label_entropia, 3, 0, 1, 1)
        
        self.label_intensidad = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_intensidad.setFont(font)
        self.label_intensidad.setAlignment(QtCore.Qt.AlignCenter)
        self.label_intensidad.setWordWrap(True)
        self.label_intensidad.setObjectName("label_intensidad")
        self.gridLayout_5.addWidget(self.label_intensidad, 1, 0, 1, 1)
        
        self.label_valor_azul = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_azul.setFont(font)
        self.label_valor_azul.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_azul.setWordWrap(True)
        self.label_valor_azul.setObjectName("label_valor_azul")
        self.gridLayout_5.addWidget(self.label_valor_azul, 0, 1, 1, 1)
        
        self.label_azul = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_azul.setFont(font)
        self.label_azul.setAlignment(QtCore.Qt.AlignCenter)
        self.label_azul.setWordWrap(True)
        self.label_azul.setObjectName("label_azul")
        self.gridLayout_5.addWidget(self.label_azul, 0, 0, 1, 1)
        
        self.label_valor_corr = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_corr.setFont(font)
        self.label_valor_corr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_corr.setWordWrap(True)
        self.label_valor_corr.setObjectName("label_valor_corr")
        self.gridLayout_5.addWidget(self.label_valor_corr, 4, 1, 1, 1)
        
        self.label_valor_intensidad = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_intensidad.setFont(font)
        self.label_valor_intensidad.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_intensidad.setWordWrap(True)
        self.label_valor_intensidad.setObjectName("label_valor_intensidad")
        self.gridLayout_5.addWidget(self.label_valor_intensidad, 1, 1, 1, 1)
        
        self.label_energia = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_energia.setFont(font)
        self.label_energia.setAlignment(QtCore.Qt.AlignCenter)
        self.label_energia.setWordWrap(True)
        self.label_energia.setObjectName("label_energia")
        self.gridLayout_5.addWidget(self.label_energia, 2, 0, 1, 1)
        
        self.label_ref_azul = QtWidgets.QLabel(self.frame_6)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_azul.setFont(font)
        self.label_ref_azul.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_azul.setWordWrap(True)
        self.label_ref_azul.setObjectName("label_ref_azul")
        self.gridLayout_5.addWidget(self.label_ref_azul, 0, 2, 1, 1)
        
        self.label_valor_ent = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_ent.setFont(font)
        self.label_valor_ent.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_ent.setWordWrap(True)
        self.label_valor_ent.setObjectName("label_valor_ent")
        self.gridLayout_5.addWidget(self.label_valor_ent, 3, 1, 1, 1)
        
        self.label_ref_ent = QtWidgets.QLabel(self.frame_6)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_ent.setFont(font)
        self.label_ref_ent.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_ent.setWordWrap(True)
        self.label_ref_ent.setObjectName("label_ref_ent")
        self.gridLayout_5.addWidget(self.label_ref_ent, 3, 2, 1, 1)
        
        self.label_ref_corr = QtWidgets.QLabel(self.frame_6)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_corr.setFont(font)
        self.label_ref_corr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_corr.setWordWrap(True)
        self.label_ref_corr.setObjectName("label_ref_corr")
        self.gridLayout_5.addWidget(self.label_ref_corr, 4, 2, 1, 1)
        
        self.label_corr = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_corr.setFont(font)
        self.label_corr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_corr.setWordWrap(True)
        self.label_corr.setObjectName("label_corr")
        self.gridLayout_5.addWidget(self.label_corr, 4, 0, 1, 1)
        
        self.label_valor_en = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_valor_en.setFont(font)
        self.label_valor_en.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valor_en.setWordWrap(True)
        self.label_valor_en.setObjectName("label_valor_en")
        self.gridLayout_5.addWidget(self.label_valor_en, 2, 1, 1, 1)
        
        self.label_ref_intensidad = QtWidgets.QLabel(self.frame_6)
        # font = QtGui.QFont()
        # font.setPointSize(8)
        # font.setBold(True)
        # font.setWeight(75)
        # self.label_ref_intensidad.setFont(font)
        self.label_ref_intensidad.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ref_intensidad.setWordWrap(True)
        self.label_ref_intensidad.setObjectName("label_ref_intensidad")
        self.gridLayout_5.addWidget(self.label_ref_intensidad, 1, 2, 1, 1)
        
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.verticalLayout_5.addWidget(self.frame_4)
        self.verticalLayout_2.addWidget(self.frame_estadistica)
        self.verticalLayout.addWidget(self.frame_cuerpo)
        MainWindow.setCentralWidget(self.centralwidget)

        self.pushButton.clicked.connect(self.load_image)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_nombre.setText(_translate("MainWindow", "Inserte imagen de la lesión que desea clasificar:"))
        self.pushButton.setText(_translate("MainWindow", "Cargar imagen"))
        self.label_8.setText(_translate("MainWindow", "Original"))
        #self.label_imgOR.setText(_translate("MainWindow", "TextLabel"))
        #self.label_imgSeg.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "Segmentada"))
        self.label_2.setText(_translate("MainWindow", "Probabilidad de que sea maligno:"))
        #self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_irregularidad.setText(_translate("MainWindow", "Irregularidad del borde:"))
        self.label_ref_cant.setText(_translate("MainWindow", "Benignas: 1.0\nMalignas: 2.0"))
        self.label_ref_circu.setText(_translate("MainWindow", "Benignas: 72.32\nMalignas: 89.34"))
        self.label_ref_asimetria.setText(_translate("MainWindow", "Referencias\nBenignas: 26.08\nMalignas: 37.69"))
        self.label_circulo.setText(_translate("MainWindow", "Desviación del círculo:"))
        #self.label_valor_bordes.setText(_translate("MainWindow", "TextLabel"))
        self.label_cantcolores.setText(_translate("MainWindow", "Cantidad de colores:"))
        #self.label_valor_cant.setText(_translate("MainWindow", "que"))
        self.label_bordes.setText(_translate("MainWindow", "Bordes internos:"))
        self.label_ref_bordes.setText(_translate("MainWindow", "Benignas: 0.13\nMalignas: 0.18"))
        #self.label_valor_irr.setText(_translate("MainWindow", "TextLabel"))
        self.label_ref_irr.setText(_translate("MainWindow", "Benignas: 16.68\nMalignas: 20.36"))
        #self.label_valor_asimetria.setText(_translate("MainWindow", "TextLabel"))
        #self.label_valor_circ.setText(_translate("MainWindow", "TextLabel"))
        self.label_asimetria.setText(_translate("MainWindow", "Asimetría:"))
        self.label_ref_en.setText(_translate("MainWindow", "Benignas: 0.0074\nMalignas: 0.0095"))
        self.label_entropia.setText(_translate("MainWindow", "Entropía:"))
        self.label_intensidad.setText(_translate("MainWindow", "Intesidad de la capa azul:"))
        #self.label_valor_azul.setText(_translate("MainWindow", "TextLabel"))
        self.label_azul.setText(_translate("MainWindow", "Azul maligno:"))
        #self.label_valor_corr.setText(_translate("MainWindow", "que"))
        #self.label_valor_intensidad.setText(_translate("MainWindow", "TextLabel"))
        self.label_energia.setText(_translate("MainWindow", "Energía:"))
        self.label_ref_azul.setText(_translate("MainWindow", "Referencias\nBenignas: no presenta\nMalignas: presenta"))
        #self.label_valor_ent.setText(_translate("MainWindow", "TextLabel"))
        self.label_ref_ent.setText(_translate("MainWindow", "Benignas: 7.23\nMalignas: 6.89"))
        self.label_ref_corr.setText(_translate("MainWindow", "Benignas: 1.33e-09\nMalignas: -6.46e-12"))
        self.label_corr.setText(_translate("MainWindow", "Correlación:"))
        #self.label_valor_en.setText(_translate("MainWindow", "TextLabel"))
        self.label_ref_intensidad.setText(_translate("MainWindow", "Benignas: 121.74\nMalignas: 87.47"))

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly

        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Seleccionar imagen", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")

        if file_path:
            self.image = cv2.imread(file_path,1)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_azul=self.image[:,:,2]
            self.label_imgOR.display_image(self.image, False)
            self.process_image()

        return

    def scientific_notation(self, number):
        return f'{number:.3e}'

    def segmentacion_asimetria(self):
        #otsu
        umbral1, otsu = cv2.threshold(self.image_azul,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #morfologia
        #cierre
        kernel = np.ones((3, 3), 'uint8')
        binaria_inv = (otsu == 0).astype('uint8')
        binaria_inv = binaria_inv.astype("uint8")
        binaria_dil= cv2.dilate(binaria_inv, kernel, iterations=1)
        binaria_dil_eros = cv2.erode(binaria_dil, kernel, iterations=1)
        #apertura
        binaria_dil_eros_eros= cv2.erode(binaria_dil_eros, kernel, iterations=1)
        binaria_dil_eros_eros_dil = cv2.dilate(binaria_dil_eros_eros, kernel, iterations=1)
        self.binaria=binaria_dil_eros_eros_dil

    def asimetria (self):
        #bounding box
        cnts = cv2.findContours(self.binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        imagen2 = self.image_azul.copy()
        imagen2_color=self.image.copy()
        original = imagen2.copy()
        h_max=0
        w_max=0
        y_max=0
        x_max=0
        alto_im, ancho_im = self.image_azul.shape
        dif_centro=10**10
        #me quedo con el bounding box mas grande y que este mas centrado
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if dif_centro>math.sqrt((alto_im/2-(y+h/2))**2+(ancho_im/2-(x+w/2))**2):
                if h*w > h_max*w_max and h>=20 and w>=20:
                    h_max=h
                    w_max=w
                    x_max=x
                    y_max=y
                    dif_centro=math.sqrt((alto_im/2-(y+w/2))**2+(ancho_im/2-(x+h/2))**2)
            #cv2.rectangle(imagen2, (x, y), (x + w, y + h), (36,255,12), 2) # parámetros, img, punto inicial, punto final, color, espesor
        self.ROI = original[y_max:y_max+h_max, x_max:x_max+w_max]
        self.ROI_color = imagen2_color[y_max:y_max+h_max, x_max:x_max+w_max]



        #calculo la media de intensidad de todo el lunar
        media_intensidad = cv2.mean(self.ROI)[0]

        #recorto la imagen en cuadrados

        # Obtiene las dimensiones de la ROI
        alto, ancho = self.ROI.shape

        # Tamaño de los cuadrados
        tamano_cuadrado_alto = alto//20
        tamano_cuadrado_ancho = ancho//20

        # Inicializa una lista para almacenar los cuadrados
        cuadrados = []
        medias = []
        imagen_resultante=np.zeros(self.ROI.shape)
        imagen_resultante2=np.zeros(self.ROI.shape)
        # Recorre la imagen en pasos de 12x12
        for i in range(0, alto, tamano_cuadrado_alto):
            for j in range(0, ancho, tamano_cuadrado_ancho):
                cuadrado = self.ROI[i:i + tamano_cuadrado_alto, j:j + tamano_cuadrado_ancho]
                media = int(np.mean(cuadrado))
                for k in range (cuadrado.shape[0]):
                    for l in range (cuadrado.shape[1]):
                        imagen_resultante[i+k, j+l]=media
                        if media>=media_intensidad:
                            imagen_resultante2[i+k, j+l]=255

                cuadrados.append(cuadrado)
                medias.append(media)


        #dividimos en 4 cuadrantes y calculamos la cantidad de cuadrados con 0 en cada cuadrante
        mitad_ancho=ancho//2
        mitad_alto=alto//2

        #primer cuadrante
        cuadrante1=0
        for i in range(0, mitad_alto, tamano_cuadrado_alto):
            for j in range(0, mitad_ancho, tamano_cuadrado_ancho):
                if imagen_resultante2[i+tamano_cuadrado_alto//2][j+tamano_cuadrado_ancho//2] == 0:
                    cuadrante1+=1


        #segundo cuadrante
        cuadrante2=0
        for i in range(0, mitad_alto, tamano_cuadrado_alto):
            for j in range(mitad_ancho, ancho-(tamano_cuadrado_ancho), tamano_cuadrado_ancho):
                if imagen_resultante2[i+tamano_cuadrado_alto//2][j+tamano_cuadrado_ancho//2] == 0:
                    cuadrante2+=1


        #tercer cuadrante
        cuadrante3=0
        for i in range(mitad_alto, alto-tamano_cuadrado_alto, tamano_cuadrado_alto):
            for j in range(0, mitad_ancho, tamano_cuadrado_ancho):
                if imagen_resultante2[i+tamano_cuadrado_alto//2][j+tamano_cuadrado_ancho//2]== 0:
                    cuadrante3+=1


        #cuarto cuadrante
        cuadrante4=0
        for i in range(mitad_alto, alto-tamano_cuadrado_alto, tamano_cuadrado_alto):
            for j in range(mitad_ancho, ancho-(tamano_cuadrado_ancho), tamano_cuadrado_ancho):
                if imagen_resultante2[i+tamano_cuadrado_alto//2][j+tamano_cuadrado_ancho//2]== 0:
                    cuadrante4+=1

        #coeficiente de asimetria
        self.coef_asimetria=math.sqrt((cuadrante1+cuadrante2-(cuadrante3+cuadrante4))**2+(cuadrante1+cuadrante3-(cuadrante2+cuadrante4))**2)

    def circularidad (self):

        #binarizo con otsu cada ROI
        umbral, otsu = cv2.threshold(self.ROI,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binaria_inv = (otsu==0)
        self.binaria_inv = binaria_inv.astype("uint8")

        #calculo la cantidad de pixeles blancos que hay en la binarizada = Area
        area=len(np.where(self.binaria_inv==1)[0])

        #calculo el borde de canny sobre la imagen binarizada
        canny = cv2.Canny(self.binaria_inv, 1, 1)

        #cuento la cantidad de pixeles blancos que hay en los bordes canny = perimetro
        perimetro = len(np.where(canny==255)[0])

        #calculo el centro de masa
        x_array=np.where(self.binaria_inv==1)[1]
        xcm=sum(x_array)/len(x_array)
        y_array=np.where(self.binaria_inv==1)[0]
        ycm=sum(y_array)/len(y_array)

        #calculo la distancia de los bordes al cm
        min_dist=10**10
        max_dist=0
        x_border=np.where(canny==255)[1]
        y_border=np.where(canny==255)[0]
        dist_euc=[]
        for i in range(len(x_border)):
            dist_euc.append(math.sqrt((xcm-x_border[i])**2+(ycm-y_border[i])**2))

        #maximas y minimas distancias
        max_dist=np.max(dist_euc)
        min_dist=np.min(dist_euc)

        #desviacion estandar de las distancias al cm
        dist_prom=np.mean(dist_euc)
        suma=0
        for i in range(len(dist_euc)):
            suma+=(dist_euc[i]-dist_prom)**2

        self.desviacion_circulo=math.sqrt((1/(len(dist_euc)-1))*suma)

        self.irregularidad_bordes=perimetro/max_dist

        ROI_area = self.binaria_inv*self.ROI
        ROI_ec = cv2.equalizeHist(ROI_area)
        canny2 = cv2.Canny(ROI_ec, 100, 120)

        self.bordes_internos = (len(np.where(canny2==255)[0])-perimetro)/(area)


    def matriz_de_co_ocurrencia (self):
        co_occurrence_matrix = cv2.calcHist([self.ROI], [0], None, [256], [0, 256])
        co_occurrence_matrix /= co_occurrence_matrix.sum()
        return co_occurrence_matrix

    def energia(self, mat):
        return np.sum(mat**2)

    def entropia(self, mat):
        g_non_zero = mat[mat > 0]
        return -np.sum(g_non_zero * np.log2(g_non_zero))

    def correlacion(self, mat):
        rows, cols = mat.shape
        mu = sum(i * mat[i, j] for i in range(rows) for j in range(cols))
        sigma = sum((i - mu) ** 2 * mat[i, j] for i in range(rows) for j in range(cols))
        return sum((i - mu) * (j - mu) * mat[i, j] for i in range(rows) for j in range(cols)) / sigma

    def texturas (self):
        matriz = self.matriz_de_co_ocurrencia ()
        self.valor_energia = self.energia(matriz)
        self.valor_entropia = self.entropia(matriz)
        self.valor_corr = self.correlacion(matriz)

    def cantidad_tot(self):
        rangos = [ #BGR !!!!!!!!!!!!!!!!
        (np.array([200, 200, 200], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8)),
        (np.array([185, 180, 180], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8)),
        (np.array([180, 115, 180], dtype=np.uint8), np.array([255, 180, 255], dtype=np.uint8)),
        (np.array([0, 0, 0], dtype=np.uint8), np.array([100, 115, 125], dtype=np.uint8)),
        (np.array([0, 50, 153], dtype=np.uint8), np.array([100, 153, 198], dtype=np.uint8)),
        (np.array([0, 0, 50], dtype=np.uint8), np.array([100, 170, 153], dtype=np.uint8)),
        (np.array([100, 100, 0], dtype=np.uint8), np.array([180, 125, 153], dtype=np.uint8))
        ]
        umbral = 1.5
        colores = []
        
        for rango in rangos:
            mask = cv2.inRange(self.imagen_color_fondonegro, rango[0], rango[1])

            # Contar píxeles en cada máscara y porcentaje
            count = np.count_nonzero(mask)
            porcentaje = (count / (self.imagen_color_fondonegro.shape[0] * self.imagen_color_fondonegro.shape[1])) * 100

            if porcentaje >= umbral:
                colores.append(porcentaje)
                
        self.cantidad_colores = len(colores)

    def azulgris(self):
        rango_color_bajo1 = np.array([148, 123, 148], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto1 = np.array([152, 127, 152], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!

        rango_color_bajo2 = np.array([148, 123, 123], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto2 = np.array([152, 127, 127], dtype=np.uint8)

        rango_color_bajo3 = np.array([123, 98, 98], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto3 = np.array([127, 102, 102], dtype=np.uint8)

        rango_color_bajo4 = np.array([148, 123, 98], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto4 = np.array([152, 127, 102], dtype=np.uint8)

        rango_color_bajo5 = np.array([148, 98, 48], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto5 = np.array([151, 102, 102], dtype=np.uint8)

        rango_color_bajo6 = np.array([148, 98, 0], dtype=np.uint8) #BGR !!!!!!!!!!!!!!!!
        rango_color_alto6 = np.array([152, 102, 2], dtype=np.uint8)

        # Crear una máscara para el rango de colores
        mask1 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo1, rango_color_alto1)
        mask2 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo2, rango_color_alto2)
        mask3 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo3, rango_color_alto3)
        mask4 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo4, rango_color_alto4)
        mask5 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo5, rango_color_alto5)
        mask6 = cv2.inRange(self.imagen_color_fondonegro, rango_color_bajo6, rango_color_alto6)

        # Cuenta la cantidad de píxeles que cumplen con el rango
        pixeles_cumplen_rango = np.count_nonzero(mask1) + np.count_nonzero(mask2) + np.count_nonzero(mask3) + np.count_nonzero(mask4) + np.count_nonzero(mask5) + np.count_nonzero(mask6)
        #print(pixeles_cumplen_rango)

        # Calcula el porcentaje de píxeles que cumplen con el rango con respecto al total de píxeles no negros
        porcentaje = (pixeles_cumplen_rango / (self.imagen_color_fondonegro.shape[0] * self.imagen_color_fondonegro.shape[1])) * 100
        #print(porcentaje)

        if porcentaje >= 0.5:
            self.azul_blanquecino = 1
        else:
            self.azul_blanquecino = 0 


    def process_image(self):
        #Aca iria el procesamiento!
        #segmetacion
        self.segmentacion_asimetria()
        #asimetria y ROI
        self.asimetria() #genero self.coef_asimetria y self.ROI
        self.circularidad() #genero self.irregularidad_bordes, self.desviacion_circulo, self.bordes_internos
        
        # Separar las capas RGB de la imagen original
        r, g, b = cv2.split(self.ROI_color)

        # Multiplicar cada capa por la imagen binarizada
        r_multiplicada = r * self.binaria_inv
        g_multiplicada = g * self.binaria_inv
        b_multiplicada = b * self.binaria_inv

        # Combinar las capas multiplicadas
        self.imagen_color_fondonegro = cv2.merge((r_multiplicada, g_multiplicada, b_multiplicada))

        self.label_imgSeg.display_image(self.imagen_color_fondonegro, False)
        
        self.cantidad_tot()
        self.azulgris()
        self.texturas()
        
        self.promedio_intensidad = np.mean(self.ROI_color[:,:,2])

        #parametros
        self.label_valor_asimetria.setText(str(round(self.coef_asimetria,3)))
        self.label_valor_circ.setText(str(round(self.desviacion_circulo,3)))
        self.label_valor_irr.setText(str(round(self.irregularidad_bordes,3)))
        self.label_valor_bordes.setText(str(round(self.bordes_internos,3)))

        self.label_valor_en.setText(self.scientific_notation((self.valor_energia)))
        self.label_valor_ent.setText(str(round(self.valor_entropia,3)))
        self.label_valor_corr.setText(self.scientific_notation((self.valor_corr)))
        self.label_valor_cant.setText(str(self.cantidad_colores))
        if self.azul_blanquecino == 1:
            self.label_valor_azul.setText('Presenta')
        else:
            self.label_valor_azul.setText('No presenta')

        self.label_valor_intensidad.setText(str(round(self.promedio_intensidad,3)))

        print(self.valor_entropia)
        print(self.valor_corr)
        print(self.bordes_internos)

        #Ejemplo de como tiene que quedar (estan inventados los numeros)
        asimetria=3.62490000e+01
        coef_irregularidad=9.43835500e+00
        desv_circulo=1.00060032e+02
        bordes=1.03744000e-01
        cant_colores=1.00000000e+00
        azul_m=0.00000000e+00
        icp=1.00416269e+02
        energia= self.valor_energia
        entropia= self.valor_entropia
        correlacion= self.valor_corr

        #Llamar a la funcion ML con todos los parametros, los 4 de cata y los de barbie ya los puse con los nombres bien
        #Pasar en este orden a la funcion
        self.MachineLearning(self.coef_asimetria,self.irregularidad_bordes,self.desviacion_circulo,self.bordes_internos,self.cantidad_colores,self.azul_blanquecino,self.promedio_intensidad,self.valor_energia,self.valor_entropia,self.valor_corr)



    def MachineLearning(self,asimetria,coef_irregularidad,desv_circulo,bordes,cant_colores,azul_maligno,int_capa_azul,energia,entropia,correlacion):
        parametros=[asimetria,coef_irregularidad,desv_circulo,bordes,cant_colores,azul_maligno,int_capa_azul,energia,entropia,correlacion]
        prediction = modelo.predict_proba([parametros])[:, 1]
        probabilidad=round(prediction[0]*100,3)
        self.label_7.setText(str(probabilidad)+'%') #display en el label

        return


#Para visualizar las imagenes con matplotlib
class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super(MatplotlibWidget, self).__init__(fig)
        self.setParent(parent)
        self.ax.axis("off")

    def display_image(self, image, gris):
        # Display the image using Matplotlib
        self.ax.clear()
        if gris == True:
            self.ax.imshow(image, cmap='gray')
        else:
            self.ax.imshow(image)
        self.ax.axis("off")
        self.draw()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
