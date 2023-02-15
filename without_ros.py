import pylibfreenect2



import numpy as np
from PIL import Image as ima
import imageio as iio
import cv2 
import math
import torch

class yolo:
    def __init__(self,FX_DEPTH,FY_DEPTH,CX_DEPTH,CY_DEPTH,image,depth):
        self.FX_DEPTH = FX_DEPTH
        self.FY_DEPTH = FY_DEPTH
        self.CX_DEPTH = CX_DEPTH
        self.CY_DEPTH = CY_DEPTH
        self.image = image
        self.depth = depth
    def distance(self,xmin,ymin,xmax,ymax,depth):
        cx=(xmax+xmin)/2
        cy=(ymax+ymin)/2
        z = depth[int(cx)][int(cy)]
        x = (int(cy) - self.CX_DEPTH) * z / self.FX_DEPTH
        y = (int(cx) - self.CY_DEPTH) * z / self.FY_DEPTH
        tong=x**2+y**2+z**2
        distance = math.sqrt(tong)
        distance=distance/1000
        distance = round(distance,2)
        return distance
    def draw_boundingbox(self,xmin,xmax,ymin,ymax,distance):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 1
        org = (int(xmin),int(ymin))
        c1 = (int(xmin),int(ymin))
        cv2.rectangle(image, c1, (int(((xmax+xmin)/2)+xmax-xmin),int(((ymax+ymin)/2)+ymax-ymin)),color,4)
        image = cv2.putText(image,str(distance)+'m', org, font, fontScale, color, thickness, cv2.LINE_AA)
    def mediapipe(self):
        YOLO=yolo()
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False    
        result = yolo_model(img)    
        img.flags.writeable = True   
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        MARGIN=10
        for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist():
            distance = YOLO.distance(xmin, ymin, xmax,ymax,self.depth)
            YOLO.draw_boundingbox(self,xmin,xmax,ymin,ymax,distance)
            cv2.imshow('cam_kinect',img)
            cv2.waitKey(1)   
if __name__ == '__main__':
    # camera_parameter
    FX_DEPTH = 1081.3720703125
    FY_DEPTH = 1081.3720703125
    CX_DEPTH = 959.5
    CY_DEPTH = 539.5
    IMAGE_WIDTH=600
    IMAGE_HEIGHT=500
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo_model.classes=[0]

    YOLO = yolo(FX_DEPTH,FY_DEPTH,CX_DEPTH,CY_DEPTH,image,depth)
    YOLO.mediapipe()
    // load Camera Kinect
