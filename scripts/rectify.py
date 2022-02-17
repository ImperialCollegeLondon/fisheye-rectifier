#!/usr/bin/env python
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo

camera_params = {}
bridge = CvBridge()

# load yaml via opencv
def load_yaml(filepath):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    
    camera_params["height"] = fs.getNode("height").real()
    camera_params["width"] = fs.getNode("width").real()

    camera_params["K1"] = fs.getNode("K1").mat()
    camera_params["K2"] = fs.getNode("K2").mat()
    
    camera_params["D1"] = fs.getNode("D1").mat().flatten()[:4]
    camera_params["D2"] = fs.getNode("D2").mat().flatten()[:4]

    camera_params["R1"] = fs.getNode("R1").mat()
    camera_params["R2"] = fs.getNode("R2").mat()

    camera_params["P1"] = fs.getNode("P1").mat()
    camera_params["P2"] = fs.getNode("P2").mat()


class Rectification(object):
    def __init__(self):

        camera_name = rospy.get_param("~t265_cam_namespace", default="/t265")

        self.left_image_pub = rospy.Publisher(camera_name + "/fisheye1/image_rect", Image, queue_size=1)
        self.left_caminfo_pub = rospy.Publisher(camera_name + "/fisheye1/rect/camera_info", CameraInfo, queue_size=1)
        self.right_image_pub = rospy.Publisher(camera_name + "/fisheye2/image_rect", Image, queue_size=1)
        
        left_cam_sub = message_filters.Subscriber(camera_name + "/fisheye1/image_raw", Image)
        right_cam_sub = message_filters.Subscriber(camera_name + "/fisheye2/image_raw", Image)
        # Create distortion maps for left and right
        image_size = (int(camera_params["width"]), int(camera_params["height"]))
        m1type = cv2.CV_16SC2
        undistortion_map_1 = cv2.fisheye.initUndistortRectifyMap(
            camera_params["K1"], 
            camera_params["D1"],
            camera_params["R1"],
            camera_params["P1"],
            image_size,
            m1type)

        undistortion_map_2 = cv2.fisheye.initUndistortRectifyMap(
            camera_params["K2"], 
            camera_params["D2"], 
            camera_params["R2"], 
            camera_params["P2"], 
            image_size, 
            m1type)
        
        self.undistort_rectify = { "left": undistortion_map_1, "right": undistortion_map_2 }

        ts = message_filters.TimeSynchronizer([left_cam_sub, right_cam_sub], 10)
        ts.registerCallback(self.camera_cb)

        self.counter = 0

    def build_camera_info(self):
        camera_info = CameraInfo()

        width = 848
        height = 800

        camera_info.header.seq = self.counter
        self.counter += 1
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = "t265_fisheye1_optical_frame"
        
        camera_info.width = width
        camera_info.height = height
        camera_info.distortion_model = 'plumb_bob'

        camera_info.K = [width, 0.0, width / 2.0, 0.0, height, height / 2.0, 0.0, 0.0, 1.0]
        camera_info.D = [0, 0, 0, 0, 0]
        camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        camera_info.P = [width, 0.0, width / 2.0, 0.0, 0.0, height, height / 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        return camera_info 


    def cv_to_ros(self, image, message):
        image_message = bridge.cv2_to_imgmsg(image, encoding="rgb8")
        image_message.header = message.header
        return image_message

    def camera_cb(self, img_left, img_right):
        cv_left = bridge.imgmsg_to_cv2(img_left, desired_encoding='rgb8')
        cv_right = bridge.imgmsg_to_cv2(img_right, desired_encoding='rgb8')
        cv_left_undistorted = cv2.remap(src=cv_left, map1=self.undistort_rectify["left"][0], map2=self.undistort_rectify["left"][1], interpolation=cv2.INTER_LINEAR)
        cv_right_undistorted = cv2.remap(src=cv_right, map1=self.undistort_rectify["right"][0], map2=self.undistort_rectify["right"][1], interpolation=cv2.INTER_LINEAR)
        self.left_image_pub.publish(self.cv_to_ros(cv_left_undistorted, img_left))
        self.right_image_pub.publish(self.cv_to_ros(cv_right_undistorted, img_right))
        self.left_caminfo_pub.publish(self.build_camera_info())

def main():
    rospy.init_node('rectification', anonymous=False)
    rectify = Rectification()
    while not rospy.is_shutdown():
        rospy.sleep(5.0)

if __name__ == '__main__':
    try:
        load_yaml("/root/ros_ws/src/fisheye-rectifier/params/t265_params.yml")
        main()
    except rospy.ROSInterruptException:
        pass
