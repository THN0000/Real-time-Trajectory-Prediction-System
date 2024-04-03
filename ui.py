import numpy as np
import tracker
from detector_CPU import Detector
import cv2
from matplotlib.pyplot import MultipleLocator
import matplotlib; matplotlib.use('TkAgg')
import torch
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from frets import Model
import matplotlib.pyplot as plt
import yaml
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def draw_bboxes(image, bboxes, line_thickness,pointlist,blue,green):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4

    # 遍历所有的 人
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        # 撞线的点
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        #cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        xx = int((x1 + x2)/2)
        yy = int((y1 + y2)/2)
        pointlist[int(pos_id)].append([xx,yy])
        blue.append([xx,yy])
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        #cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    #[225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        list_pts.clear()

    return image,pointlist,blue,green

def draw_cross(img,x,y,color):
    cv2.line(img, (x - 5, y + 5),
             (x + 5, y - 5), color, 3)
    cv2.line(img, (x - 5, y - 5),
             (x + 5, y + 5), color, 3)
    # print('indraw_cross')
    return img

def det_yolov7(info1):
    # 初始化
    # 读取配置文件
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # 打印读取到的配置参数
    print("Config Parameters:")
    print(config)
    trajectory = Model(config)
    trajectory.load_state_dict(torch.load('./weights/raw.pth'))

    detector = Detector()
    pointlist = [[] for i in range(100)]
    pointcopy = [[] for i in range(100)]
    # 打开视频
    print('info1:',info1)
    capture = cv2.VideoCapture(info1)

    count = 0
    rr = 0
    # H = np.array([
    #     [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
    #     [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
    #     [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
    # ])
    H = np.array([
        [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
        [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
        [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]
    ])

    H_inv = np.linalg.inv(H)
    blue = []
    green = []

    while True:
        # 读取每帧图片
        ret, im = capture.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 显示帧
        # cv2.imshow('Video', frame)
        count += 1
        if im is None:
            print('break')
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)
        # 如果画面中 有bbox

        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0

            output_image_frame, pointlist, blue, green = draw_bboxes(im, list_bboxs, line_thickness=1,
                                                                     pointlist=pointlist, blue=blue, green=green)
            if rr == 1:
                for poo in green:
                    cv2.circle(im, (int(poo[0]), int(poo[1])), 3, [0, 255, 0], -1)

            # 遍历每个人
            for ii in range(len(pointlist)):
                # 如果这个人不为空
                if pointlist[ii] != []:
                    for po in range(len(pointlist[ii])):
                        if rr == 1:
                            color0 = [0, 0, 255]
                        else:
                            color0 = [255, 0, 0]
                        cv2.circle(im, (pointlist[ii][po][0], pointlist[ii][po][1]), 3, color0, -1)
                        if po == len(pointlist[ii]) - 1:
                            im = draw_cross(im, pointlist[ii][po][0], pointlist[ii][po][1], color0)

            for ii in range(len(pointcopy)):
                # 如果这个人不为空
                if pointcopy[ii] != []:
                    # 遍历每个坐标
                    for po in pointcopy[ii]:
                        cv2.circle(im, (int(po[0]), int(po[1])), 3, (255, 0, 0), -1)

            if count >= 30:
                rr = 1
                count = 0

                green = []
                # 开启预测
                # 遍历每个人# 遍历每个人的轨迹，并查看每个轨迹的形状
                for idx, person_traj in enumerate(pointlist):
                    print(f"Person {idx + 1} trajectory shape: {len(person_traj)}")
                    if len(person_traj) >= 8:
                        history_coords = np.array(pointlist[idx][-8:])

                        # padded_coords = np.array(pad_or_truncate_sequence(history_coords, 8))
                        # print('padded_coords:', padded_coords)

                        # 将像素坐标转换为齐次坐标，添加一列1
                        pixel_coords_homogeneous = np.column_stack(
                            (history_coords, np.ones((history_coords.shape[0], 1))))
                        # 使用单应矩阵对齐次坐标进行变换
                        world_coords_homogeneous = np.dot(H, pixel_coords_homogeneous.T).T
                        # 将变换后的齐次坐标转换为非齐次坐标
                        world_coords = world_coords_homogeneous[:, :2] / world_coords_homogeneous[:, 2:]


                        history_coords_tensor = torch.tensor(world_coords, dtype=torch.float32)
                        print('history_coords_tensor:', history_coords_tensor)

                        history_coords_tensor = history_coords_tensor.unsqueeze(0).permute(0, 2, 1)
                        # 遍历每个坐标

                        with torch.no_grad():  # 禁止梯度计算
                            trajectory.eval()  # 将模型设置为评估模式
                            future_coords_tensor = trajectory(history_coords_tensor)  # 使用模型进行预测
                            print('future_coords_tensor:', future_coords_tensor)
                        future_coords = future_coords_tensor.numpy()  # 将预测结果转换为NumPy数组

                        # 将世界坐标转换为齐次坐标，添加一列1
                        world_coords_homogeneous = np.column_stack(
                            (future_coords, np.ones((future_coords.shape[0], 1))))
                        # 使用单应矩阵逆矩阵对齐次坐标进行变换
                        pixel_coords_homogeneous = np.dot(H_inv, world_coords_homogeneous.T).T
                        # 将变换后的齐次坐标转换为非齐次坐标
                        pixel_coords = pixel_coords_homogeneous[:, :2] / pixel_coords_homogeneous[:, 2:]
                        print('pixel_coords:', pixel_coords)
                        for pixel_coord in pixel_coords:
                            # 检查点是否在图像范围内
                            if check_inside_image(pixel_coord, im.shape):
                                # 在图像中绘制像素坐标点
                                # cv2.circle(im, (int(pixel_coord[0]), int(pixel_coord[1])), 3, (0, 255, 0), -1)
                                green.append([int(pixel_coord[0]), int(pixel_coord[1])])
                for po in green:
                    print(po)
                    cv2.circle(im, (int(po[0]), int(po[1])), 3, (0, 255, 0), -1)
                ui.showimg(im)
                QApplication.processEvents()



                for jj in range(50):
                    for ii in range(len(pointlist)):
                                # 如果这个人不为空
                        if pointlist[ii] != []:
                            print(len(pointlist[ii]))
                            if jj < len(pointlist[ii]):
                                detax = pointlist[ii][-1][0] - pointlist[ii][0][0]
                                detay = pointlist[ii][-1][1] - pointlist[ii][0][1]
                                print(detay)
                                cv2.circle(im, (pointlist[ii][jj][0] + detax, pointlist[ii][jj][1] + detay), 3,
                                           (0, 255, 0), -1)

                                if detay < -50:
                                    detay = int(detay * 1.2)
                                if detay < 0 and detay > -50:
                                    detay = int(detay * 1.1)
                                if detay > 50:
                                    detay = int(detay * 1.2)
                                if detay > 0 and detay < 50:
                                    detay = int(detay * 1.1)
                                green.append([pointlist[ii][jj][0] + detax, pointlist[ii][jj][1] + detay])
                    ui.showimg(im)
                    QApplication.processEvents()
                pointcopy = pointlist.copy()
                # 清除列表
                pointlist = [[] for i in range(100)]
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass
        ui.showimg(output_image_frame)
        QApplication.processEvents()
        pass
    pass

    capture.release()
    cv2.destroyAllWindows()


def pad_or_truncate_sequence(sequence, length):
    if len(sequence) < length:
        # 填充操作：将序列填充到指定长度
        padded_sequence = sequence + [[0, 0]] * (length - len(sequence))
        return padded_sequence
    elif len(sequence) > length:
        # 截断操作：将序列截断为指定长度
        truncated_sequence = sequence[:length]
        return truncated_sequence
    else:
        return sequence

# 检查点是否在图像范围内
def check_inside_image(point, image_shape):
    height, width = image_shape[:2]
    return 0 <= point[0] < height and 0 <= point[1] < width

class Thread_1(QThread):  # 线程1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov7(info1)




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/background.jpg\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 40, 1200, 840))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.3);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(100, 900, 200, 40))
        self.pushButton.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(400, 900, 200, 40))
        self.pushButton_3.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")



        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1000, 900, 200, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "实时轨迹预测系统"))
        self.label_2.setText(_translate("MainWindow", "实时轨迹预测系统"))
        self.pushButton.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_3.setText(_translate("MainWindow", "开启预测"))
        self.pushButton_2.setText(_translate("MainWindow", "退出系统"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_3.clicked.connect(self.click_1)
        self.pushButton_2.clicked.connect(self.handleCalc2)

    def handleCalc2(self):
        os._exit(0)

    def openfile(self):
        global sname,filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        global duration
        if filepath[-4:] == '.mp4':
            print(filepath, sname)
            cap = cv2.VideoCapture(filepath)
            # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
            # get方法参数按顺序对应下表（从0开始编号)
            rate = cap.get(5)  # 帧速率
            FrameNumber = cap.get(7)  # 视频文件的帧数
            duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟





    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 1200
        else:
            ratio = n_height / 1200
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        print(img2.shape[1],img2.shape[0])
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        global blue,green
        blue = []
        green = []
        try:
            self.thread_1.quit()
        except:
            pass
        try:
            self.thread_1 = Thread_1(filepath)  # 创建线程
        except:
            self.thread_1 = Thread_1('0')  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程

    # def click_1(self):
    #     global blue, green
    #     blue = []
    #     green = []
    #     try:
    #         self.thread_1.quit()
    #     except:
    #         pass
    #     try:
    #         self.thread_1 = Thread_1(filepath)  # 创建线程
    #     except:
    #         self.thread_1 = Thread_1('0')  # 创建线程
    #     self.thread_1.start()  # 启动线程


class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Welcome to Login')  # 设置标题
        self.resize(600, 500)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮
        self.setStyleSheet("background-image: url(\"./template/1.png\")")

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move(185, 180)

        # self.verticalLayout = QVBoxLayout(self.frame)
        self.mainLayout = QVBoxLayout(self.frame)

        # self.nameLb1 = QLabel('&Name', self)
        # self.nameLb1.setFont(QFont('Times', 24))
        self.nameEd1 = QLineEdit(self)
        self.nameEd1.setFixedSize(150, 30)
        self.nameEd1.setPlaceholderText("account number")
        # 设置透明度
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        # 设置文本框为圆角
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # self.nameLb1.setBuddy(self.nameEd1)


        self.nameEd3 = QLineEdit(self)
        self.nameEd3.setPlaceholderText("password")
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')

        self.btnOK = QPushButton('Sign in')
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')  # font-family中可以设置字体大小，如下font-size:24px;

        self.btnCancel = QPushButton('register')
        op4 = QGraphicsOpacityEffect()
        op4.setOpacity(1)
        self.btnCancel.setGraphicsEffect(op4)
        self.btnCancel.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')

        # self.btnOK.setFont(QFont('Microsoft YaHei', 24))
        # self.btnCancel.setFont(QFont('Microsoft YaHei', 24))

        # self.mainLayout.addWidget(self.nameLb1, 0, 0)
        self.mainLayout.addWidget(self.nameEd1)

        # self.mainLayout.addWidget(self.nameLb2, 1, 0)

        self.mainLayout.addWidget(self.nameEd3)

        self.mainLayout.addWidget(self.btnOK)
        self.mainLayout.addWidget(self.btnCancel)

        self.mainLayout.setSpacing(50)


        # 绑定按钮事件
        self.btnOK.clicked.connect(self.button_enter_verify)
        self.btnCancel.clicked.connect(self.button_register_verify)  # 返回按钮绑定到退出

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("login was successful")


    def button_enter_verify(self):
        # 校验账号是否正确
        global administrator, userstext, passtext
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        users = os.listdir(path1)

        for i in users:
            with open(path1 + '/' + i, "r") as f:
                userstext.append(i[:-4])
                passtext.append(f.readline())

        for i in users:
            if i[:-4] == self.nameEd1.text():
                with open(path1 + '/' + i, "r") as f:
                    if f.readline() == self.nameEd3.text():
                        if i[:2] == 'GM':
                            administrator = 1
                            self.accept()
                        else:
                            passtext.append(f.readline())
                            self.accept()
                    else:
                        self.nameEd3.setText("Password error")
                        pw = 1
        if pw == 0:
            self.nameEd1.setText("Account error")



if __name__ == "__main__":
    # 创建应用
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    # login_ui = LoginDialog()
    # 校验是否验证通过
    # if login_ui.exec_() == QDialog.Accepted:
    # 初始化主功能窗口
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    # 设置应用退出
    sys.exit(window_application.exec_())

