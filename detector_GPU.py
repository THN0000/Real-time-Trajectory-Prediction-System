import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.1
        self.stride = 1

        # self.weights = './weights/attention_mechanism.pt'
        self.weights = './weights/output_of_small_target_detection.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):
        print('in detect')
        # cv2.waitKey(10)
        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        print(pred)
        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['bicycle','car', 'bus', 'truck']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    xm = x2
                    ym = y2
                    #if ym-1.86*xm+375.72<0 and ym-1.23*xm+362.85>0:   #上行下行车道
                    #if  ym - 1.23 * xm + 362.85 < 0: #垂直进入
                    #if ym - 1.51 * xm + 377.5 < 0 and ym - 1.23 * xm + 362.85 > 0: #上行车道
                    if  ym +0.797* xm -509.77 > 0:
                       boxes.append(
                            (x1, y1, x2, y2, lbl, conf))

        return boxes
