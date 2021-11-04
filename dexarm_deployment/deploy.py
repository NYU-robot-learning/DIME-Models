# Standard imports
import numpy as np
import cv2

# Standard ROS imports
import rospy
from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError

# Dexterous aarm control package import
from move_dexarm import DexArmControl

# Importing the Allegro IK library
from allegro_ik import AllegroInvKDL

# Model Imports
from model_scripts.inn import INNDeploy

# Other imports
from copy import deepcopy as copy

# ROS Topics to get the AR Marker data
IMAGE_TOPIC = "/cam_1/color/image_raw"
MARKER_TOPIC = "/visualization_marker"
JOINT_STATE_TOPIC = "/allegroHand_0/joint_states"

class DexArmDeploy():
    def __init__(self, debug = False, model = "inn", device = "cpu"):
        # Initializing ROS deployment node
        try:
            rospy.init_node("model_deploy")
        except:
            pass

        # Setting the debug parameter
        self.debug = debug

        # Initializing arm controller
        self.arm = DexArmControl()

        # Initializing Allegro Ik Library
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/urdf_template/allegro_right.urdf")

        # Initializing the model
        if model == "inn":
            if self.debug is True:
                self.model = INNDeploy(k = 1, load_image_data = True, device = device)
            else:
                self.model = INNDeploy(k = 1, device = device)
        # Moving the dexterous arm to home position
        self.arm.home_robot()

        # Initializing topic data
        self.image = None
        self.allegro_joint_state = None
        self.ar_marker_data = None

        # Other initializations
        self.bridge = CvBridge()

        rospy.Subscriber(MARKER_TOPIC, Marker, self._callback_ar_marker_data, queue_size = 1)
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        rospy.Subscriber(IMAGE_TOPIC, Image, self._callback_image, queue_size=1)

    def _callback_ar_marker_data(self, data):
        self.ar_marker_data = data

    def _callback_joint_state(self, data):
        self.allegro_joint_state = data

    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def get_cube_position(self):
        cube_position = np.array([self.ar_marker_data.pose.position.x, self.ar_marker_data.pose.position.y, self.ar_marker_data.pose.position.z])
        return cube_position

    def get_tip_coords(self):
        thumb_coord = self.allegro_ik.finger_forward_kinematics('thumb', list(self.allegro_joint_state.position)[12:16])[0]
        ring_coord = self.allegro_ik.finger_forward_kinematics('ring', list(self.allegro_joint_state.position)[8:12])[0]

        return thumb_coord, ring_coord

    def update_joint_state(self, thumb_tip_coord, ring_tip_coord):
        current_joint_angles = list(self.allegro_joint_state.position)

        thumb_joint_angles = self.allegro_ik.finger_inverse_kinematics('thumb', thumb_tip_coord, current_joint_angles[12:16])
        ring_joint_angles = self.allegro_ik.finger_inverse_kinematics('ring', ring_tip_coord, current_joint_angles[8:12])

        desired_joint_angles = copy(current_joint_angles)
        
        for idx in range(4):
            desired_joint_angles[8 + idx] = ring_joint_angles[idx]
            desired_joint_angles[12 + idx] = thumb_joint_angles[idx]

        return desired_joint_angles

    def deploy(self):
        while True:
            # Wating for key
            next_step = input()

            # Setting the break condition
            if next_step == "q":
                break

            print("Getting state data...")

            # Getting the state data
            thumb_tip_coord, ring_tip_coord = self.get_tip_coords()
            cube_pos = self.get_cube_position()

            print("Current state data:\n Thumb-tip position: {}\n Ring-tip position: {}\n Cube position: {}\n".format(thumb_tip_coord, ring_tip_coord, cube_pos))

            if self.debug is True:
                action, nn_state_image_path = self.model.get_action_with_image(thumb_tip_coord, ring_tip_coord, cube_pos)
                action = list(action.reshape(6))

                print("Action len: {}".format(action))

                # Reading the Nearest Neighbor images 
                nn_state_image = cv2.imread(nn_state_image_path)

                combined_images = np.concatenate((cv2.resize(self.image, (640,360), interpolation = cv2.INTER_AREA), cv2.resize(nn_state_image, (640,360), interpolation = cv2.INTER_AREA)), axis=1)
                cv2.imshow("Current state and Nearest Neighbor Images", combined_images)
                cv2.waitKey(1)
            else:
                action = list(self.model.get_action(thumb_tip_coord, ring_tip_coord, cube_pos))

                print("Action len: {}".format(len(action)))

            print("Corresponding action: ", action)

            updated_thumb_tip_coord = np.array(thumb_tip_coord) + np.array(action[0:3])
            updated_ring_tip_coord = np.array(ring_tip_coord) + np.array(action[3:6])

            desired_joint_angles = self.update_joint_state(updated_thumb_tip_coord, updated_ring_tip_coord)

            print("Moving arm to {}\n".format(desired_joint_angles))
            self.arm.move_hand(desired_joint_angles)