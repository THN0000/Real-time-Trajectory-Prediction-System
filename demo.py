import numpy as np
import tracker
from detector_CPU import Detector
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
    return img

if __name__ == "__main__":
    # 初始化
    detector = Detector()
    pointlist = [[] for i in range(100)]
    pointcopy = [[] for i in range(100)]
    # 打开视频
    capture = cv2.VideoCapture('./2.avi')
    count = 0
    rr = 0
    blue = []
    green = []
    while True:
        # 读取每帧图片
        _, im = capture.read()
        count += 1
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        # Mouse.mouse(im)
        # cv2.waitKey(0)

        list_bboxs = []
        bboxes = detector.detect(im)
        # 如果画面中 有bbox

        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0

            output_image_frame, pointlist,blue,green = draw_bboxes(im, list_bboxs, line_thickness=1,pointlist=pointlist,blue = blue,green = green)
            if rr == 1:
                for poo in green:
                    cv2.circle(im, (poo[0], poo[1]), 3, [0, 255, 0], -1)


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
                        cv2.circle(im, (po[0], po[1]), 3, (255, 0, 0),-1)

            if count > 50:
                rr = 1
                count = 0

                green = []
                # 开启预测
                # 遍历每个人
                for jj in range(50):
                    for ii in range(len(pointlist)):
                        # 如果这个人不为空
                        if pointlist[ii] != []:
                            # 遍历每个坐标
                            if jj < len(pointlist[ii]):
                                detax = pointlist[ii][-1][0] - pointlist[ii][0][0]
                                detay = pointlist[ii][-1][1] - pointlist[ii][0][1]
                                print(detay)
                                cv2.circle(im, (pointlist[ii][jj][0] + detax, pointlist[ii][jj][1] + detay), 3, (0, 255, 0), -1)

                                if detay < -50:
                                    detay = int(detay*1.2)
                                if detay < 0 and detay > -50:
                                    detay = int(detay*1.1)
                                if detay > 50:
                                    detay = int(detay*1.2)
                                if detay > 0 and detay < 50:
                                    detay = int(detay*1.1)
                                green.append([pointlist[ii][jj][0] + detax,pointlist[ii][jj][1] + detay])
                    cv2.imshow('frame', im)
                    cv2.waitKey(100)
                pointcopy = pointlist.copy()
                # 清除列表
                pointlist = [[] for i in range(100)]
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass
        cv2.imshow('frame',output_image_frame)
        cv2.waitKey(1)
        pass
    pass

    capture.release()
    cv2.destroyAllWindows()


