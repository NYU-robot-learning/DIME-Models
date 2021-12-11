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
from ik_teleop.ik_core.allegro_ik import AllegroInvKDL

# Model Imports
from model_scripts.inn import INNDeploy

# Other imports
from copy import deepcopy as copy

# ROS Topics to get the AR Marker data
IMAGE_TOPIC = "/cam_1/color/image_raw"
MARKER_TOPIC = "/visualization_marker"
JOINT_STATE_TOPIC = "/allegroHand_0/joint_states"

class DexArmDeploy():
    def __init__(self, k = 1, use_abs = True, debug = False, cube_priority = 1, device = "cpu"):
        # Initializing ROS deployment node
        try:
            rospy.init_node("model_deploy")
        except:
            pass

        # Setting the debug parameter only if k is 1
        if k == 1:
            self.debug = debug
        else:
            self.debug = False

        self.abs = use_abs

        # Initializing arm controller
        self.arm = DexArmControl()

        # Initializing Allegro Ik Library
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/ik_teleop/urdf_template/allegro_right.urdf")

        # Initializing INN
        if self.debug is True:
            self.model = INNDeploy(k = 1, load_image_data = True, cube_priority = cube_priority, device = device)
        else:
            self.model = INNDeploy(k = k, cube_priority = cube_priority, device = device)

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
        index_coord = self.allegro_ik.finger_forward_kinematics('index', list(self.allegro_joint_state.position)[:4])[0]
        middle_coord = self.allegro_ik.finger_forward_kinematics('middle', list(self.allegro_joint_state.position)[4:8])[0]
        ring_coord = self.allegro_ik.finger_forward_kinematics('ring', list(self.allegro_joint_state.position)[8:12])[0]
        thumb_coord = self.allegro_ik.finger_forward_kinematics('thumb', list(self.allegro_joint_state.position)[12:16])[0]

        return index_coord, middle_coord, ring_coord, thumb_coord

    def update_joint_state(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord):
        current_joint_angles = list(self.allegro_joint_state.position)

        index_joint_angles = self.allegro_ik.finger_inverse_kinematics('index', index_tip_coord, current_joint_angles[0:4])
        middle_joint_angles = self.allegro_ik.finger_inverse_kinematics('middle', middle_tip_coord, current_joint_angles[4:8])
        ring_joint_angles = self.allegro_ik.finger_inverse_kinematics('ring', ring_tip_coord, current_joint_angles[8:12])
        thumb_joint_angles = self.allegro_ik.finger_inverse_kinematics('thumb', thumb_tip_coord, current_joint_angles[12:16])

        desired_joint_angles = copy(current_joint_angles)
        
        for idx in range(4):
            desired_joint_angles[idx] = index_joint_angles[idx]
            desired_joint_angles[4 + idx] = middle_joint_angles[idx]
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

            print("Getting state data...\n")

            print("Current joint angles: {}\n".format(self.allegro_joint_state.position))

            # Getting the state data
            index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord = self.get_tip_coords()
            cube_pos = self.get_cube_position()

            print("Current state data:\n Index-tip position: {}\n Middle-tip position: {}\n Ring-tip position: {}\n Thumb-tip position: {}\n Cube position: {}\n".format(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, cube_pos))

            if self.debug is True:
                action, nn_state_image_path, trans_state_image_path, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, cube_l2_diff = self.model.get_action_with_image(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, cube_pos)
                action = list(action.reshape(12))

                print("Trans state img path: {}".format(trans_state_image_path))

                print("Distances:\n Index distance: {}\n Middle distance: {}\n Ring distance: {}\n Thumb distance: {}\n Cube distance: {}".format(index_l2_diff.item(), middle_l2_diff.item(), ring_l2_diff.item(), thumb_l2_diff.item(), cube_l2_diff.item()))
                
                # Reading the Nearest Neighbor images 
                nn_state_image = cv2.imread(nn_state_image_path)
                trans_state_image = cv2.imread(trans_state_image_path)

                # Combing the surrent state image and NN image and printing them
                combined_images = np.concatenate((
                    cv2.resize(self.image, (420, 240), interpolation = cv2.INTER_AREA), 
                    cv2.resize(nn_state_image, (420, 240), interpolation = cv2.INTER_AREA),
                    cv2.resize(trans_state_image, (420, 240), interpolation = cv2.INTER_AREA)
                ), axis=1)

                cv2.imshow("Current state and Nearest Neighbor Images", combined_images)
                cv2.waitKey(1)

            else:
                action = list(self.model.get_action(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, cube_pos).reshape(12))

            print("Corresponding action: ", action)

            if self.abs:
                # print("Using absolute values for thumb: {} and for ring finger: {}".format(action[:3], action[3:]))
                updated_index_tip_coord = action[0:3]
                updated_middle_tip_coord = action[3:6]
                updated_ring_tip_coord = action[6:9]
                updated_thumb_tip_coord = action[9:12]
            else:
                updated_index_tip_coord = np.array(index_tip_coord) + np.array(action[0:3])
                updated_middle_tip_coord = np.array(middle_tip_coord) + np.array(action[3:6])
                updated_ring_tip_coord = np.array(ring_tip_coord) + np.array(action[6:9])
                updated_thumb_tip_coord = np.array(thumb_tip_coord) + np.array(action[9:12])

            desired_joint_angles = self.update_joint_state(updated_index_tip_coord, updated_middle_tip_coord, updated_ring_tip_coord, updated_thumb_tip_coord)

            print("Moving arm to {}\n".format(desired_joint_angles))
            self.arm.move_hand(desired_joint_angles)


if __name__ == "__main__":
    d = DexArmDeploy(debug = False, k = 3, cube_priority = 2, device = "cuda")
    d.deploy()