# -*- coding: utf-8 -*-
# Код создан студентом группы САПР-2.1п Харитоновым Анатолием Александровичем
from PyQt5 import QtCore, QtGui, QtWidgets
from window import Ui_MainWindow
from dialog import Ui_Dialog
from result import Ui_Result
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from SkipgramModeler import load_skip, train_skip, get_skip_cosine
from CBOWModeler import load_cbow, train_cbow, get_cbow_cosine
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing.dummy import Pool as ThreadPool 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
import linkgrammar.clinkgrammar as clg
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from pyspark.sql import SQLContext
from pyspark import SparkContext
from segmentation import segment
import matplotlib.pyplot as plt
from functools import partial
from lxml import etree
import numpy as np
import pymorphy2
import datetime
import random
import math
import sys
import re
import os

class Dialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())

class Result(QtWidgets.QDialog, Ui_Result):
    def __init__(self, data):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.tableWidget.setRowCount(len(data))
        for ind, el in enumerate(data):
            self.tableWidget.setItem(ind, 0, QtWidgets.QTableWidgetItem(el[0]))
            self.tableWidget.setItem(ind, 1, QtWidgets.QTableWidgetItem(el[1]))
            self.tableWidget.setItem(ind, 2, QtWidgets.QTableWidgetItem(el[2]))
            self.tableWidget.setItem(ind, 3, QtWidgets.QTableWidgetItem(el[3]))
            self.tableWidget.setItem(ind, 4, QtWidgets.QTableWidgetItem(el[4]))
            self.tableWidget.setItem(ind, 5, QtWidgets.QTableWidgetItem(el[5]))
            self.tableWidget.setItem(ind, 6, QtWidgets.QTableWidgetItem(el[6]))
        self.tableWidget.resizeRowsToContents()

class App(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.all_patent = []
        self.need_patent = []
        self.model = None 
        self.model_type = 0
        self.word_to_ix = []
        self.tfidf = []
        self.idf = []
        self.dgh = {}
        self.patent = {}
        self.labels = []
        self.res_data = []
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.open_dir_btn.clicked.connect(self.open_dir)
        self.train_w2v_btn.clicked.connect(self.train_w2v)
        self.extract_dgh_btn.clicked.connect(self.extract_dgh)
        self.help_btn.clicked.connect(self.help)
        self.result_btn.clicked.connect(self.result)
        self.addToolBar(NavigationToolbar(self.img_wdg.canvas, self))

    def result(self):
        if (self.res_data == []):
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Нет извлеченных технических функций.")
            return
        res = Result(self.res_data)
        res.show()
        result = res.exec_()

    def open_dir(self):  
        """
        Выбор директории и открытие хмл файлов в ней
        """
        self.all_patent.clear()
        self.need_patent.clear()
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, 'Select directory', '/home')
        if (dir_name == ""):
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Директория не выбрана")
            return
        message = datetime.datetime.now().time().strftime("%X") + " - начат поиск патентов в дирректории.\n"
        self.log_txt.setText(message) 
        self.centralwidget.repaint()
        files = get_list_patent(dir_name)
        pool = ThreadPool(8) 
        pool.map(self.is_patent, files)
        pool.close() 
        pool.join()
        message += datetime.datetime.now().time().strftime("%X") + " - поиск окончен.\n"
        message += "Всего найдено " + str(len(files)) + " xml файлов в выбранной дериктории.\n"
        message += "Из них " + str(len(self.all_patent)) + " РосПатентов.\n"
        message += "Среди которых " + str(len(self.need_patent)) + " патентов из секции E F G или H."
        self.log_txt.setText(message) 
        if self.need_patent == []: 
            return
        def sort_col(i):
            return i[2]
        self.need_patent.sort(key = sort_col)
        first_date = self.need_patent[0][2]
        last_date = self.need_patent[-1][2]
        self.date_hsd.setMinimum(int(first_date))
        self.date_hsd.setMaximum(int(last_date))
        self.date_hsd.setValue(int(last_date))
        first_date = first_date[6:8] + "." + first_date[4:6] + "." + first_date[0:4]
        last_date = last_date[6:8] + "." + last_date[4:6] + "." + last_date[0:4]
        self.first_date_lbl.setText(first_date)
        self.last_date_lbl.setText(last_date)

    def train_w2v(self):
        if len(self.need_patent) == 0:
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Патенты не выбраны!")
            return
        dlg = Dialog()
        dlg.show()
        result = dlg.exec_()
        if result == QtWidgets.QDialog.Accepted:
            message = datetime.datetime.now().time().strftime("%X") + " - обучение модели начато.\n"
            self.log_txt.setText(message) 
            self.centralwidget.repaint()
            if dlg.radioButton.isChecked():
                self.model, self.word_to_ix = train_cbow(self.need_patent)
                self.model_type = 1
            else:
                self.model, self.word_to_ix = train_skip(self.need_patent)
                self.model_type = 2

        if self.model == None:
            message += datetime.datetime.now().time().strftime("%X") + " - обучение завершено с ошибкой.\n"
        else:
            message += datetime.datetime.now().time().strftime("%X") + " - обучение завершено успешно.\n"
        self.log_txt.setText(message) 

    def extract_dgh(self):
        if len(self.need_patent) == 0:
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Патенты не выбраны!")
            return

        if self.model == None:
            dlg = Dialog()
            dlg.show()
            result = dlg.exec_()
            if result == QtWidgets.QDialog.Accepted:
                if dlg.radioButton.isChecked():
                    self.model, self.word_to_ix = load_cbow()
                    self.model_type = 1
                else:
                    self.model, self.word_to_ix = load_skip()
                    self.model_type = 2
                self.model = Word2Vec.load("word2vec.model")
        
        if self.model == None:
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Конфигурации модели не найдены!")
            return

        message = datetime.datetime.now().time().strftime("%X") + " - извлечение технических функций начато.\n"
        self.log_txt.setText(message) 
        self.centralwidget.repaint()
        pool = ThreadPool(8) 
        data = pool.map(self.open_xml, self.need_patent)
        pool.close() 
        pool.join()

        ix_to_dgh = {}
        ix_to_pat = {}
        i_pat = 0
        i_dgh = 0
        line = ''
        with open('some_pat.txt', 'w') as file:
            for el in data:
                flag = False
                for dgh in el[1]:
                    if dgh != {}:
                        ix_to_dgh[i_dgh] = dgh
                        i_dgh = i_dgh + 1
                        flag = True
                        if dgh.get('d') != '' and dgh.get('g') != '' and dgh.get('h') != '':
                            line = str(el[3]) + '\t' + dgh.get('d').get('main') + '\t' + dgh.get('g').get('main') + '\t' + dgh.get('h').get('main') + '\n'
                            file.write(line)
                    if flag == True:
                        ix_to_pat[i_pat] = [el[0], el[2]]
                        i_pat = i_pat + 1
        self.dgh = ix_to_dgh
        self.patent = ix_to_pat
        print(i_pat)
        print(i_dgh)
        if i_dgh == 0:
            QtWidgets.QMessageBox.about(self.centralwidget, "Ошибка", "Технические функции не найдены!")
            return
        mas_pat = [[0] * i_dgh for i in range(i_pat)]
        for i in range(i_pat):
            mas_pat[i][i] = 1
        message += datetime.datetime.now().time().strftime("%X") + " - технические функции извлечены, начат расчет TFIDF.\n"
        self.log_txt.setText(message) 
        self.centralwidget.repaint()
        self.idf, self.tfidf = self.get_tfidf(mas_pat, ix_to_dgh)
        message += datetime.datetime.now().time().strftime("%X") + " - расчет TFIDF окончен, начато построение графика.\n"
        self.log_txt.setText(message) 
        self.centralwidget.repaint()
        self.update_graph(self.tfidf)
        message += datetime.datetime.now().time().strftime("%X") + " - расчеты завершены.\n"
        self.log_txt.setText(message) 
        self.res_data.clear()
        trends = self.trend_of_year()
        first_year = int(self.patent.get(0)[1][:4])
        count_years = int(self.patent.get(len(self.tfidf)-1)[1][:4]) - first_year + 1
        predict = self.prediction(trends)
        with open('some_mark.txt', 'w') as file:
            for ind, el in enumerate(self.tfidf):
                pat_date = str(self.patent.get(ind)[1])
                pat_date = pat_date[6:8] + "." + pat_date[4:6] + "." + pat_date[0:4]
                sum_year = 0
                for n, i in enumerate(trends):
                    sum_year += i[ind] * (n + 1) / count_years
                line = str(self.idf[ind]) + '\t' +  str(sum_year) + '\t' +  str(predict[ind]) + '\t' + str(self.idf[ind] + sum_year + predict[ind]) + '\n'
                file.write(line)
                self.res_data.append([self.patent.get(ind)[0], 
                                 pat_date, 
                                 'D - ' + self.dgh.get(ind).get('d').get('main') + '\nG - ' + self.dgh.get(ind).get('g').get('main') + '\nH - ' + self.dgh.get(ind).get('h').get('main'),
                                 str(self.idf[ind]),
                                 str(sum_year),
                                 str(predict[ind]),
                                 str(self.idf[ind] + sum_year + predict[ind])])
        res = Result(self.res_data)
        res.show()
        result = res.exec_()

    def help(self):   
        """
        Вывод окна со справкой
        """
        QtWidgets.QMessageBox.about(self.centralwidget, 
            "Справка", 
            '\nДанная программа выполнена студентом группы САПР-2.1п Харитоновым Анатолием Александровичем в качестве выпускной работы магистра\n\n'
            'Список операций:\n'
            '1) При нажатии кнопки "Выбрать патентный массив" осуществляется поиск патентов в выбранной директории;\n'
            '2) При нажатии кнопки "Обучить word2vec" на выбранных патентах происходит обучение модели word2vec, необходимой для определения синонимов;\n'
            '3) При нажатии кнопки "Извлечь техн. функции" на выбранных патентах происходит исзвлечение технических функций и получение их критериальных оценок;\n'
            '4) При нажатии кнопки "Справка" открывается окно со справкой;\n'
            '5) При нажатии кнопки "Просмотреть результаты", выводится окно со всеми оценками;\n'
            '6) Ползунком можно изменить временные промежутки для которых выполняется анализ;\n'
            '7) При нажатии кнопки "Сохранить изображение" схема кластеризации сохраняется в отдельный файл.')
    
    def update_graph(self, tfidf):
        count = len(tfidf) // 15
        if count < 3:
            num_clusters = 3
        elif count > 20:
            num_clusters = 20
        else:
            num_clusters = count
        num_seeds = 10
        max_iterations = 300
        labels_color_map = {
            0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49',
            10: '#660f5a', 11: '#4f304b', 12: '#340042', 13: '#0886cf', 14: '#1c875c',
            15: '#0fa835', 16: '#19a80f', 17: '#42a80f', 18: '#a89e0f', 19: '#995c00'
        }
        pca_num_components = 2
        tsne_num_components = 2
        clustering_model = KMeans(
            n_clusters = num_clusters,
            max_iter = max_iterations,
            precompute_distances = "auto",
            n_jobs = -1
        )
        np_tfidf = clustering_model.fit_transform(tfidf)
        self.labels = clustering_model.fit_predict(tfidf)
        reduced_data = PCA(n_components=pca_num_components).fit_transform(np_tfidf)
        self.img_wdg.canvas.axes.clear()
        for index, instance in enumerate(reduced_data):
            pca_comp_1, pca_comp_2 = reduced_data[index]
            color = labels_color_map[self.labels[index]]
            self.img_wdg.canvas.axes.scatter(pca_comp_1, pca_comp_2, c = color)
        self.img_wdg.canvas.axes.set_title('Кластеризация технических функций')
        self.img_wdg.canvas.draw()

    def is_patent(self, file_name):
        file_data = []
        with open(file_name) as fobj:
            xml = fobj.read()
            root = etree.fromstring(xml.encode('utf-8'))
            if (xml.find("ru-patent-document") and root.find(".//B190") is not None):
                file_data.append(file_name)
                file_data.append(extract_number(xml))
                file_data.append(extract_date(xml))
                file_data.append(extract_name(xml))
                self.all_patent.append(file_data)
                if (root.find(".//section") is not None):
                    section = root.find(".//section").text
                if (root.find(".//class") is not None):
                    class_patent = root.find(".//class").text
                if section == "E" or section == "F" or section == "G" or section == "H":
                    self.need_patent.append(file_data)
    
    def get_tfidf(self, matrix, voc_dgh):
        model = self.model
        new_mas = []
        numSlices = len(matrix) // 8
        if numSlices < 8:
            inputRdd = sc.parallelize(matrix)
        else:
            inputRdd = sc.parallelize(matrix, numSlices = numSlices)
        tf = inputRdd.map(lambda x: compare_pat(model, x, voc_dgh))
        for x in tf.collect():
            new_mas.append(x)
        np_ar = np.array(new_mas)
        np_90 = np_ar.swapaxes(0, 1)
        pool = ThreadPool(8) 
        data = pool.map(calc_tfidf, np_90)
        pool.close() 
        pool.join()
        data = np.array(data).swapaxes(0, -1).tolist()
        tfidf = data[1]
        idf = data[0]
        return idf, np.array(tfidf).tolist()

    def open_xml(self, file_data):
        sents = []
        dghs = []
        with open(file_data[0]) as fobj:
            if file_data[2] > str(self.date_hsd.value()):
                return [file_data[3], [{}]]
            xml = fobj.read()
            root = etree.fromstring(xml.encode('utf-8'))
            descs = extract_desc(xml)
            for desc in descs:
                if desc != []:
                    sents = segment(desc).split("\n")
                    # for sent in sents:
                    dghs.append(extract_func(sents[0]))
        return [file_data[3], dghs, file_data[2], file_data[0]]

    def trend_of_year(self):
        max_iterations = 300
        first_year= int(self.patent.get(0)[1][:4])
        last_year = int(self.patent.get(len(self.tfidf)-1)[1][:4])
        top_of_date = []
        years = {}
        predict = {}
        year = int(self.patent.get(len(self.tfidf)-1)[1][:4])
        for i in range(first_year, last_year + 1):
            years[str(i)] = []
        for i in range(len(self.tfidf)):
            pat = self.patent.get(i)
            if pat[1][:4] in years.keys():
                years[pat[1][:4]].append(i)
        for key in years.keys():
            el = [0] * len(self.tfidf)
            if years[key] == []:
                top_of_date.append(el)
            elif len(years[key]) == 1:
                arr = np.nonzero(np.array(self.tfidf[years.get(key)[0]]))
                for ind in arr:
                    el[ind[0]] = 1
                top_of_date.append(el)
            else:
                count = len(years[key]) // 15
                if count < 2:
                    num_clusters = 2
                elif count > 20:
                    num_clusters = 20
                else:
                    num_clusters = count
                clustering_model = None
                clustering_model = KMeans(
                    n_clusters = num_clusters,
                    max_iter = max_iterations,
                    precompute_distances = "auto",
                    n_jobs = -1
                )
                tfidf = []
                for ind in years.get(key):
                    tfidf.append(self.tfidf[ind])
                np_tfidf = clustering_model.fit_transform(tfidf)
                labels = clustering_model.fit_predict(tfidf)
                test = clustering_model.score(tfidf)
                count_lbl = {}
                for i in labels:
                    if i in count_lbl:
                        count_lbl[i] += 1
                    else:
                        count_lbl[i] = 1
                sort_count = []
                sort_count = list(count_lbl.values())
                sort_count.sort()
                for i_dgh in range(len(self.tfidf)):
                    lbl = 0
                    val = 0
                    for i_pat in range(len(tfidf)):
                        if tfidf[i_pat][i_dgh] > val:
                            val = tfidf[i_pat][i_dgh]
                            lbl = labels[i_pat] + 1
                        elif tfidf[i_pat][i_dgh] != 0 and tfidf[i_pat][i_dgh] == val:
                            if lbl > labels[i_pat] + 1:
                                lbl = labels[i_pat] + 1
                    if lbl != 0:
                        el[i_dgh] = sort_count.index(count_lbl.get(lbl - 1)) + 1
                        el[i_dgh] = el[i_dgh] * val / self.idf[i_dgh]
                    else:
                        el[i_dgh] = 0
                top_of_date.append(el)
        return top_of_date

    def prediction(self, trends):
        predict = []
        first_year = int(self.patent.get(0)[1][:4])
        count_years = int(self.patent.get(len(self.tfidf)-1)[1][:4]) - first_year + 1
        x = []
        if count_years < 15:
            for i in range(first_year - 15 + count_years, first_year + count_years):
                x.append(i)
        else:
            for i in range(first_year, first_year + count_years):
                x.append(i)
        x = np.array(x).reshape(-1, 1)
        y = []
        if count_years < 15:
            for i in range(15 - count_years):
                y.append([0]*len(trends[0]))
        y.extend(trends)
        y = np.array(y)
        model = LinearRegression().fit(x, y)
        x_test = []
        for i in range(first_year + count_years , first_year + count_years + 10):
            x_test.append(i)
        x_test = np.array(x_test).reshape(-1, 1)
        y = np.mean(y, axis = 1).tolist()
        y_test = np.mean(model.predict(x_test), axis = 1).tolist()
        y.extend(y_test)
        model = ARIMA(y, order=(5,1,0))
        model_fit = model.fit(disp=0)
        for ind, el in enumerate(self.tfidf):
            pred_val = []
            last_ind = 0
            if len(trends) > 5:
                last_ind = len(trends) - 6
            for n, i in enumerate(trends):
                if n >= last_ind:
                    pred_val.append(i[ind])
            model_predict = model.predict(pred_val)
            model_predict = math.fabs(model_predict[-10])
            predict.append(model_predict)
        return predict

def calc_tfidf(line):
    zero = (line == 0).sum()
    log = math.log(line.size / (line.size - zero))
    return [log, log * line]

def compare_pat(model, list_dgh, voc_dgh):
    ind = list_dgh.index(1)
    for i, val in enumerate(list_dgh):
        if i != ind:
            list_dgh[i] = compare_dgh(model, voc_dgh.get(ind), voc_dgh.get(i))
    return list_dgh

def compare_dgh(model, dgh1, dgh2):
    marks = []
    for key in dgh1.keys():
        tmp1 = dgh1.get(key).get('main')
        tmp2 = dgh2.get(key).get('main')
        if tmp1 != '' and tmp2 != '':
            if (tmp1 == tmp2):
                mark = 1
            else:
                # mark = get_cosine(model, self.word_to_ix, tmp1, tmp2)
                try:
                    a = model.wv[tmp1].reshape(1, -1)
                    b = model.wv[tmp2].reshape(1, -1)
                    mark = cosine_similarity(a, b)[0][0]
                except Exception:
                    mark = 0
                if mark < 0.5:
                    mark = 0
        else:
            mark = 0
        marks.append(mark)
        if key == 'd':
            continue
        tmp1c = dgh1.get(key).get('atr')
        tmp2c = dgh2.get(key).get('atr')
        childs = []
        for el1 in tmp1c:
            mark_c = 0
            for el2 in tmp2c:
                if (el1 == el2):
                    childs.append(1)
                    break
                else:
                    # mark_tmp = get_cosine(model, self.word_to_ix, el1, el2)
                    try:
                        a = model.wv[el1].reshape(1, -1)
                        b = model.wv[el2].reshape(1, -1)
                        mark_tmp = cosine_similarity(a, b)[0][0]
                    except Exception:
                        mark_tmp = 0
                    if mark_tmp > mark_c:
                        mark_c = mark_tmp
            if mark_c < 0.5:
                childs.append(0)
            else:
                childs.append(mark_c)
        if len(childs) > 0:
            marks.append(sum(childs) / len(childs))
        else:
            marks.append(0)
    res = (marks[0] + (marks[1] + marks[2]) / 2 + (marks[3] + marks[4]) / 2) / 3
    return res

def get_list_patent(path_patent):
    """
    Поиск хмл файлов
    Аргументы:
    path_patent - путь директории содержащей хмл файлы
    Возвращаемое значение:
    list_patent - список всех путей к файлам хмл
    """
    list_patent = []
    for file in os.listdir(path_patent):
        path = os.path.join(path_patent, file)
        if (os.path.isdir(path) == False):
            if (path.find(".xml") > 0):
                list_patent.append(path)
        else:
            list_patent += get_list_patent(path)
    return list_patent

def extract_number(data):
    """
    Поиск в тексте информации о номере патента
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    doc_num - строка с номером патента
    """
    doc_num = ""
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//B110") is not None):
        doc_num = root.find(".//B110").text
    # print("Номер патента")
    # print(doc_num)
    return doc_num

def extract_date(data):
    """
    Поиск в тексте информации о дате выдачи патента
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    date - строка с датой выдачи патента
    """ 
    date = ""
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//B220") is not None):
        for d in root.find(".//B220"):
            date = d.text
    # print("Дата выдачи")
    # print(date)
    return date

def extract_name(data):
    """
    Поиск в тексте информации о названии патента
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    doc_name - строка с названием патента
    """ 
    doc_name = ""
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//ru-b542") is not None):
        doc_name = root.find(".//ru-b542").text
    # print("Название изобретения")
    # print(doc_name)
    return doc_name

def extract_desc(data):
    """
    Поиск в тексте информации о патентной формуле изобретения
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    descs - список описаний изобретения
    """ 
    presence_dgh = ["для ", "при "]
    descs = []
    txt = ""
    sent = ""
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//description") is not None):
        for desc in root.find(".//description"):
            ind = 0
            txt = str(desc.text).lower()
            if txt.find("изобретение") != -1 and txt.find("относится") != -1:
                for el in presence_dgh:
                    if not descs:
                        ind = txt.find(el)
                        if ind != -1:
                            ind += len(el)
                            sent = txt[ind:txt.find(".")]
                            words = re.findall(r'[а-я]+\-*[а-я]+', sent)
                            for word in words:
                                morf = morph.parse(word)[0].tag.POS
                                if morf == 'GRND' or morf == 'PRTF' or morf == 'PRTS':
                                    ind = sent.find(word)
                                    if sent[ind-2] == ",":
                                        sent = sent[:ind-2]
                            first_w = re.findall('^\w+', sent)
                            if first_w == []:
                                continue
                            mor = morph.parse(first_w[0])
                            nf = mor[0].normal_form
                            tag = mor[0].tag.POS
                            if nf in vword or tag == 'VERB' or tag == 'INFN':
                                descs.append(sent)
                if not descs:
                    if txt.find("Результат") != -1:
                        ind = txt.find("в ")
                        if ind != -1:
                            ind += len(el)
                            sent = txt[ind:txt.find(".")]
                            words = re.findall(r'[а-я]+\-*[а-я]+', sent)
                            for word in words:
                                morf = morph.parse(word)[0].tag.POS
                                if morf == 'GRND' or morf == 'PRTF' or morf == 'PRTS':
                                    ind = sent.find(word)
                                    if sent[ind-2] == ",":
                                        sent = sent[:ind-2]
                            first_w = re.findall('^\w+', sent)
                            if first_w == []:
                                continue
                            mor = morph.parse(first_w[0])
                            nf = mor[0].normal_form
                            tag = mor[0].tag.POS
                            if nf in vword or tag == 'VERB' or tag == 'INFN':
                                descs.append(sent)
    return descs

def extract_func(line):
    """
    Выделение действий, объектов и условий в преложении
    Аргументы:
    line - предложение
    Возвращаемое значение:
    dgh - действие объект условие
    """ 
    actions_index = []
    objects_index = []
    conditions_index = []
    dgh = {}
    sent = clg.sentence_create(line, lang_dict)
    if (len(line) > 2 and len(line) < 190):
        clg.sentence_split(sent, parse_opts)
        num_linkages = clg.sentence_parse(sent, parse_opts)
        if (num_linkages > 0):
            linkage = select_linkage(sent, num_linkages)
            diagram = clg.linkage_print_diagram(linkage, False, 800);
            num_link = clg.linkage_get_num_links(linkage)
            for i in range(num_link):
                link_analiz(i, linkage, actions_index, objects_index, conditions_index)
            dgh['d'] = fill_func(linkage, actions_index)
            dgh['g'] = fill_func(linkage, objects_index)
            dgh['h'] = fill_func(linkage, conditions_index)
            clg.linkage_delete(linkage)
    else:
        return dgh
    
    clg.sentence_delete(sent)
    return dgh

def select_linkage(sent, num_linkages):
    """
    Поиск наиболее точной структуры предложения
    Аргументы:
    sent - предложение с пометками
    num_linkages - количество вариантов структур предложения
    Возвращаемое значение:
    linkage - структура предложения выданная парсером
    """ 
    ind = 1
    is_good = False
    is_v_or_n = False
    linkage = clg.linkage_create(0, sent, parse_opts)
    for num in range(clg.linkage_get_num_words(linkage)):
        if (clg.linkage_get_word(linkage, num).find(".v") >= 0 or clg.linkage_get_word(linkage, num).find(".n") >= 0):
            is_v_or_n = True
    while (ind < num_linkages and is_good == False):
        for i in range(clg.linkage_get_num_links(linkage)):
            if (clg.linkage_get_link_label(linkage, i).find("W") == 0):
                if (clg.linkage_get_word(linkage, clg.linkage_get_link_rword(linkage, i)).find(".v") >= 0 or clg.linkage_get_word(linkage, clg.linkage_get_link_rword(linkage, i)).find(".n") >= 0):
                    is_good = True
        if (is_good == False and is_v_or_n == True and ind < num_linkages):
            clg.linkage_delete(linkage)
            linkage = clg.linkage_create(ind, sent, parse_opts)
        ind += 1
    return linkage

def link_analiz(i, linkage, actions_index, objects_index, conditions_index):
    """
    Распределение индексов слов на дейсвие, объект и условие в зависимости от связи
    Аргументы:
    i - индекс связи в структуре предложения
    linkage - структура предложения выданная парсером
    actions_index - индексы слов действий
    objects_index - индексы слов объектов
    conditions_index - индексы слов условий
    """ 
    if (clg.linkage_get_link_label(linkage, i).find("W") == 0):
        actions_index.append(clg.linkage_get_link_rword(linkage, i))
    if (clg.linkage_get_link_label(linkage, i).find("Sp") == 0):
        actions_index.append(clg.linkage_get_link_lword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("I") == 0):
        if (len(clg.linkage_get_link_label(linkage, i)) == 1):
            actions_index.append(clg.linkage_get_link_rword(linkage, i))
        elif (clg.linkage_get_link_label(linkage, i)[1] != "I"):
            actions_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("MV") == 0 and clg.linkage_get_link_label(linkage, i)[2] != "I"):
        if (clg.linkage_get_link_lword(linkage, i) in objects_index or clg.linkage_get_link_lword(linkage, i) in actions_index):
            objects_index.append(clg.linkage_get_link_rword(linkage, i))
        elif (clg.linkage_get_link_lword(linkage, i) in conditions_index):
            conditions_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("SI") == 0):
        objects_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i) == "E"):
        conditions_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("E") == 0 and clg.linkage_get_link_label(linkage, i)[1].islower() == True):
        conditions_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("EI") == 0):
        conditions_index.append(clg.linkage_get_link_lword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("J") == 0):
        conditions_index.append(clg.linkage_get_link_rword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("A") == 0 and clg.linkage_get_link_label(linkage, i)[1].islower() == True):
        if (clg.linkage_get_link_rword(linkage, i) in objects_index):
            objects_index.append(clg.linkage_get_link_lword(linkage, i))
        elif (clg.linkage_get_link_rword(linkage, i) in conditions_index):
            conditions_index.append(clg.linkage_get_link_lword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("AXP") == 0):
        if (clg.linkage_get_link_rword(linkage, i) in objects_index):
            objects_index.append(clg.linkage_get_link_lword(linkage, i))
        elif (clg.linkage_get_link_rword(linkage, i) in conditions_index):
            conditions_index.append(clg.linkage_get_link_lword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("PI") == 0):
        if (clg.linkage_get_link_rword(linkage, i) in objects_index):
            objects_index.append(clg.linkage_get_link_lword(linkage, i))
        elif (clg.linkage_get_link_rword(linkage, i) in conditions_index):
            conditions_index.append(clg.linkage_get_link_lword(linkage, i))
    elif (clg.linkage_get_link_label(linkage, i).find("M") == 0 and clg.linkage_get_link_label(linkage, i)[1] != "V"):
        if (clg.linkage_get_link_lword(linkage, i) in conditions_index):
            conditions_index.append(clg.linkage_get_link_rword(linkage, i))
        else:
            objects_index.append(clg.linkage_get_link_rword(linkage, i))

def fill_func(linkage, indexes):
    """
    Нахождение слов в структуре предложения по индексам и удаление приписок
    Аргументы:
    linkage - структура предложения выданная парсером
    indexes - индексы слов в предложении
    Возвращаемое значение:
    string - строка слов без приписок
    """ 
    string = ""
    voc = {'main': "", 'atr': []}
    atr = []
    for i in indexes:
        tmp = str(clg.linkage_get_word(linkage, i))
        if (tmp.find("[") > 0):
            ind = tmp.find("[")
        else:
            ind = tmp.find(".")
        mor = morph.parse(tmp[:ind])[0]
        if mor.tag.POS == 'NPRO' or mor.tag.POS == 'PRED' or mor.tag.POS == 'PREP' or mor.tag.POS == 'CONJ' or mor.tag.POS == 'PRCL' or mor.tag.POS == 'INTJ':
            continue
        string = mor.normal_form
        if voc.get('main') == "":
            voc['main'] = string
        else:
            atr.append(string)
    voc['atr'] = atr
    return voc

def set_opts():
    """
    Настройка параметров парсера
    """
    clg.parse_options_set_max_null_count(parse_opts,6)
    clg.parse_options_set_display_morphology(parse_opts,0)
    clg.parse_options_set_islands_ok(parse_opts, True)
    clg.parse_options_set_linkage_limit(parse_opts,10)
    clg.parse_options_set_disjunct_cost(parse_opts, 2)

def main():
    with open('333333.txt', 'r') as file:
        for line in file:
            vword.append(line[:-1])
    set_opts()
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()

vword = []
lang_dict = clg.dictionary_create_lang("ru")
parse_opts = clg.parse_options_create()
morph = pymorphy2.MorphAnalyzer()
sc = SparkContext()
sqlContext = SQLContext(sc)

if __name__ == '__main__':
    main()
