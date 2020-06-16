# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 585)
        MainWindow.setStyleSheet("font: 12pt \"Ubuntu\";")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.open_dir_btn = QtWidgets.QPushButton(self.centralwidget)
        self.open_dir_btn.setGeometry(QtCore.QRect(10, 10, 211, 51))
        self.open_dir_btn.setStyleSheet("")
        self.open_dir_btn.setObjectName("open_dir_btn")
        self.train_w2v_btn = QtWidgets.QPushButton(self.centralwidget)
        self.train_w2v_btn.setGeometry(QtCore.QRect(240, 10, 211, 51))
        self.train_w2v_btn.setObjectName("train_w2v_btn")
        self.extract_dgh_btn = QtWidgets.QPushButton(self.centralwidget)
        self.extract_dgh_btn.setGeometry(QtCore.QRect(470, 10, 201, 51))
        self.extract_dgh_btn.setObjectName("extract_dgh_btn")
        self.help_btn = QtWidgets.QPushButton(self.centralwidget)
        self.help_btn.setGeometry(QtCore.QRect(690, 10, 201, 51))
        self.help_btn.setObjectName("help_btn")
        self.date_hsd = QtWidgets.QSlider(self.centralwidget)
        self.date_hsd.setGeometry(QtCore.QRect(10, 130, 281, 16))
        self.date_hsd.setOrientation(QtCore.Qt.Horizontal)
        self.date_hsd.setObjectName("date_hsd")
        self.result_btn = QtWidgets.QPushButton(self.centralwidget)
        self.result_btn.setGeometry(QtCore.QRect(10, 70, 211, 51))
        self.result_btn.setObjectName("result_btn")
        self.log_lbl = QtWidgets.QLabel(self.centralwidget)
        self.log_lbl.setGeometry(QtCore.QRect(10, 180, 281, 16))
        self.log_lbl.setObjectName("log_lbl")
        self.log_txt = QtWidgets.QTextBrowser(self.centralwidget)
        self.log_txt.setGeometry(QtCore.QRect(10, 200, 281, 321))
        self.log_txt.setObjectName("log_txt")
        self.first_date_lbl = QtWidgets.QLabel(self.centralwidget)
        self.first_date_lbl.setGeometry(QtCore.QRect(10, 150, 141, 16))
        self.first_date_lbl.setText("")
        self.first_date_lbl.setObjectName("first_date_lbl")
        self.first_date_lbl.setStyleSheet("font-size: 11px; qproperty-alignment: AlignLeft;")
        self.last_date_lbl = QtWidgets.QLabel(self.centralwidget)
        self.last_date_lbl.setGeometry(QtCore.QRect(180, 150, 111, 16))
        self.last_date_lbl.setText("")
        self.last_date_lbl.setObjectName("last_date_lbl")
        self.last_date_lbl.setStyleSheet("font-size: 11px; qproperty-alignment: AlignRight;")
        self.img_wdg = MplWidget(self.centralwidget)
        self.img_wdg.setGeometry(QtCore.QRect(300, 70, 601, 461))
        self.img_wdg.setStyleSheet("background-color: rgb(250, 250, 250);")
        self.img_wdg.setObjectName("img_wdg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа по получению критериальных оценок"))
        self.open_dir_btn.setText(_translate("MainWindow", "Выбрать\n"
"патентный массив"))
        self.train_w2v_btn.setText(_translate("MainWindow", "Обучить\n"
"word2vec"))
        self.extract_dgh_btn.setText(_translate("MainWindow", "Извлечь\n"
"техн. функции"))
        self.help_btn.setText(_translate("MainWindow", "Справка"))
        self.result_btn.setText(_translate("MainWindow", "Просмотреть\nрезультаты"))
        self.log_lbl.setText(_translate("MainWindow", "Лог:"))
from mplwidget import MplWidget
