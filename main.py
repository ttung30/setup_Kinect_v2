import os
import matplotlib.pyplot as plt
from PIL import Image as ima
import imageio as iio
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
IMAGE_WIDTH=600
IMAGE_HEIGHT=500
import math
import message_filters
import sys
sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')
mp_drawing = mp.solutions.drawing_utils
mp_pose =mp.solutions.pose
# PyTorch Hub
import torch

def publish_image(imgdata):
        br = CvBridge()
        image_pub.publish(br.cv2_to_imgmsg(imgdata))

def disa(xmin,ymin,xmax,ymax,depth):
        cx=(xmax+xmin)/2
        cy=(ymax+ymin)/2
        z = depth[int(cx)][int(cy)]
        x = (int(cy) - CX_DEPTH) * z / FX_DEPTH
        y = (int(cx) - CY_DEPTH) * z / FY_DEPTH
        tong=x**2+y**2+z**2
        distance = math.sqrt(tong)
        distance=distance/1000
        distance = round(distance,2)
        return distance

def mediapipe(image,depth):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #making image writeable to false improves prediction
    image.flags.writeable = False    
    result = yolo_model(image)    
    
    # Recolor image back to BGR for rendering
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(result.xyxy)  # img1 predictions (tensor)
    #This array will contain crops of images incase we need it 
    img_list =[]
    
    #we need some extra margin bounding box for human crops to be properly detected
    MARGIN=10
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 1
       
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 1
       
    for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist():
      with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        #Media pose prediction ,we are 
        results = pose.process(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
        #Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing 
        mp_drawing.draw_landmarks(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                              )
        distance =disa(xmin, ymin, xmax,ymax,depth)
        org = (int(xmin),int(ymin))
        c2 = (int(xmax),int(ymax))
        c1 = (int(xmin),int(ymin))
        cv2.rectangle(image, c1, (int(((xmax+xmin)/2)+xmax-xmin),int(((ymax+ymin)/2)+ymax-ymin)),color,4)
        image = cv2.putText(image,str(distance)+'m', org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('sdfasd',image)
        cv2.waitKey(5)
    publish_image(image) 
        
class yolo:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth=message_filters.Subscriber('/kinect2/hd/image_depth_rect', Image)
        self.rgb=message_filters.Subscriber('/kinect2/hd/image_color', Image)  
    def image_callback_1(self,rgb,depth):
        global ros_image
        global r
        br = CvBridge()
        ros_image = br.imgmsg_to_cv2(rgb)
        r = br.imgmsg_to_cv2(depth)
        with torch.no_grad():
            mediapipe(ros_image,r)
if __name__ == '__main__':
    img_cvt=yolo()
    # camera_p
    FX_DEPTH = 1081.3720703125
    FY_DEPTH = 1081.3720703125
    CX_DEPTH = 959.5
    CY_DEPTH = 539.5
    rospy.init_node('model_yolo')
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo_model.classes=[0]
    try:
        ts = message_filters.ApproximateTimeSynchronizer([img_cvt.rgb,img_cvt.depth],10,0.1)
        ts.registerCallback(img_cvt.image_callback_1)
        image_pub = rospy.Publisher('/yolo_result_out', Image, queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
