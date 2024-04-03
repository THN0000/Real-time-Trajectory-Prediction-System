import numpy as np
# import Mouse
import tracker
from detector_GPU import Detector
import cv2
from matplotlib.pyplot import MultipleLocator
import matplotlib; matplotlib.use('TkAgg')
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def detect(info1):# ./video/1.mp4
    listx = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # 初始化2个撞线polygon
    list_pts_blue = [[0, 400], [0, 780], [1920, 780], [1920, 400]]

    ndarray_pts_blue = np.array(list_pts_blue, np.int32)

    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[0,800], [0, 1080], [1920,1080], [1920, 800]]

    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)

    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 30, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    car_down_count = 0
    bus_down_count = 0
    truck_down_count = 0
    bicycle_down_count = 0
    # 离开数量
    up_count = 0
    car_up_count = 0
    bus_up_count = 0
    truck_up_count = 0
    bicycle_up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(info1)
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    num = 0
    num2 = 0
    while True:
        num += 1
        if num == 24.8 * 1 * 60:
            num2 += 1
            QApplication.processEvents()
            ui.printf('第 %d 个1分钟 车流量为 %d' % (num2, math.ceil(down_count*100/98)))
            QApplication.processEvents()
            down_count = 0
            num = 0
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))


        #Mouse.mouse(im)
        #cv2.waitKey(0)

        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1
                        QApplication.processEvents()
                        ui.printf(
                            f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')
                        QApplication.processEvents()
                        if str(label) == 'car':
                            car_up_count += 1
                        if str(label) == 'bus':
                            bus_up_count += 1
                        if str(label) == 'truck':
                            truck_up_count += 1
                        if str(label) == 'bicycle':
                            bicycle_up_count += 1
                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1
                        QApplication.processEvents()
                        ui.printf(
                            f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')
                        QApplication.processEvents()
                        if str(label) == 'car':
                            car_down_count += 1
                        if str(label) == 'bus':
                            bus_down_count += 1
                        if str(label) == 'truck':
                            truck_down_count += 1
                        if str(label) == 'bicycle':
                            bicycle_down_count += 1
                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass
        if num == 25:
            num2 += 1
            listx.append(num2)
            list1.append(car_down_count)
            list2.append(car_up_count)
            list3.append(bus_down_count)
            list4.append(bus_up_count)
            list5.append(truck_down_count)
            list6.append(truck_up_count)
            list7.append(bicycle_down_count)
            list8.append(bicycle_up_count)
            if num2 == 1:
                plt.plot(listx, list1, color="b", linestyle="-", linewidth=1, label="car down count")
                plt.plot(listx, list2, color="c", linestyle="-", linewidth=1, label="car up count")
                plt.plot(listx, list3, color="g", linestyle="-", linewidth=1, label="bus down count")
                plt.plot(listx, list4, color="k", linestyle="-", linewidth=1, label="bus up count")
                plt.plot(listx, list5, color="m", linestyle="-", linewidth=1, label="truck down count")
                plt.plot(listx, list6, color="r", linestyle="-", linewidth=1, label="truck up count")
                plt.plot(listx, list7, color="teal", linestyle="-", linewidth=1, label="bicycle_down_count")
                plt.plot(listx, list8, color="y", linestyle="-", linewidth=1, label="bicycle_up_count")
            else:
                plt.plot(listx, list1, color="b", linestyle="-", linewidth=1)
                plt.plot(listx, list2, color="c", linestyle="-", linewidth=1)
                plt.plot(listx, list3, color="g", linestyle="-", linewidth=1)
                plt.plot(listx, list4, color="k", linestyle="-", linewidth=1)
                plt.plot(listx, list5, color="m", linestyle="-", linewidth=1)
                plt.plot(listx, list6, color="r", linestyle="-", linewidth=1)
                plt.plot(listx, list7, color="teal", linestyle="-", linewidth=1)
                plt.plot(listx, list8, color="y", linestyle="-", linewidth=1)
            plt.xlabel('Time(s)')
            plt.ylabel('Traffic flow(n)')
            plt.title('Real time line chart of traffic flow')
            plt.legend()
            plt.pause(0.0001)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口
            num = 0


        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(0,0,255), thickness=2)

        QApplication.processEvents()
        ui.showimg(output_image_frame)
        QApplication.processEvents()
        #cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()

class Thread_1(QThread):  # 线程1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        detect(info1)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1113, 848)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(320, 5, 460, 60))
        self.textBrowser.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 155, 191, 51))
        self.pushButton.setStyleSheet("background-color: rgb(0,255,0);\n"
"font: 20pt \"3ds\";")
        self.pushButton.setObjectName("pushButton")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(100, 70, 901, 61))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setStyleSheet("font: 12pt \"3ds\";\n"
"background-color: rgb(253, 255, 211);")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textEdit = QtWidgets.QTextEdit(self.layoutWidget)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 228, 261, 16))
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 12pt \"3ds\";")
        self.label_3.setObjectName("label_3")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(20, 250, 261, 561))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(710, 155, 191, 51))
        self.pushButton_3.setStyleSheet("background-color: rgb(255, 0, 0);\n"
"font: 20pt \"3ds\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(290, 230, 801, 16))
        self.label_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 12pt \"3ds\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(290, 260, 781, 501))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(720, 260, 371, 301))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1113, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:35pt;\">车流量统计系统</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "开始识别"))
        self.label.setText(_translate("MainWindow", "输入路径："))
        self.label_3.setText(_translate("MainWindow", "              日志"))
        self.pushButton_3.setText(_translate("MainWindow", "停止识别"))
        self.label_5.setText(_translate("MainWindow", "                                            识别结果"))

    def handleCalc3(self):
        global video
        video.release()
        os._exit(0)

    def printf(self,text):
        self.textBrowser_2.append(text)
        self.cursor = self.textBrowser_2.textCursor()
        self.textBrowser_2.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 800
        else:
            ratio = n_height / 800
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        video.write(img2)
        print(img2.shape[1],img2.shape[0])
        self.label_6.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        info1 = self.textEdit.toPlainText()

        for line in info1.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
        self.thread_1 = Thread_1(info1)  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程

if __name__ == "__main__":
    size = (1918, 1046)
    video = cv2.VideoWriter("./Video.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, size)
    app = QtWidgets.QApplication(sys.argv)  # 创建一个QApplication，也就是你要开发的软件app
    MainWindow = QtWidgets.QMainWindow()  # 创建一个QMainWindow，用来装载你需要的各种组件、控件
    ui = Ui_MainWindow()  # ui是Ui_MainWindow()类的实例化对象
    ui.setupUi(MainWindow)  # 执行类中的setupUi方法，方法的参数是第二步中创建的QMainWindow
    MainWindow.show()  # 执行QMainWindow的show()方法，显示这个QMainWindow
    sys.exit(app.exec_())  # 使用exit()或者点击关闭按钮退出QApplicat
