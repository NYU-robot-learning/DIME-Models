# Standard imports
import numpy as np
import cv2
import torch
from PIL import Image as PILImage

# Standard ROS imports
import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError

# Dexterous aarm control package import
from move_dexarm import DexArmControl

# Importing the Allegro IK library
from ik_teleop.ik_core.allegro_ik import AllegroInvKDL

# Model Imports
from arm_models.dexarm_deployment.model_scripts.visual_inn import VINNDeploy

# Other imports
from copy import deepcopy as copy
from torchvision import models
from torchvision import transforms as T

# ROS Topics to get the AR Marker data
IMAGE_TOPIC = "/cam_1/color/image_raw"
JOINT_STATE_TOPIC = "/allegroHand/joint_states"
ENCODER_MODEL_CHKPT = "/home/sridhar/dexterous_arm/models/arm_models/dexarm_deployment/model_checkpoints/rotation/BYOL-VINN-rotation-lowest-train-loss.pth"

class DexArmVINNDeploy():
    def __init__(self, task, k = 1, use_abs = True, debug = False, device = "cpu"):
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
        print("Initializing controller!")
        self.arm = DexArmControl()

        # Initializing Allegro Ik Library
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/ik_teleop/urdf_template/allegro_right.urdf")

        # Initializing INN
        print("Initializing model!") # TODO - put data path
        if task == "rotate": 
            if self.debug is True:
                self.model = VINNDeploy(k = 1, load_image_data = True, device = device)
            else:
                self.model = VINNDeploy(k = k, device = device)
        elif task == "flip":
            if self.debug is True:
                self.model = VINNDeploy(k = 1, load_image_data = True, device = device)
            else:
                self.model = VINNDeploy(k = k, device = device)

        # Making sure the hand moves
        self.threshold = 0.02

        # Moving the dexterous arm to home position
        print("Homing robot!\n")
        self.arm.home_robot()

        # Initializing topic data
        self.image = None
        self.allegro_joint_state = None

        # Initializing the representation extractor
        self.device = torch.device(device)
        original_encoder_model = models.resnet50(pretrained = True)
        self.encoder = torch.nn.Sequential(*(list(original_encoder_model.children())[:-1]))
        self.encoder.to(self.device)
        self.encoder.load_state_dict(torch.load(ENCODER_MODEL_CHKPT))
        self.encoder.eval()

        # Initializing image transformation function
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean = torch.tensor([0.3484, 0.3638, 0.3819]), 
                std = torch.tensor([0.3224, 0.3151, 0.3166])  
            )
        ])

        # Other initializations
        self.bridge = CvBridge()

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
            if self.allegro_joint_state is None:
                print('No allegro state received!')
                continue

            if self.image is None:
                # print('No robot image received!')
                continue

            # Wating for key
            next_step = input()

            # Setting the break condition
            if next_step == "q":
                break

            print("Getting state data...\n")

            print("Current joint angles: {}\n".format(self.allegro_joint_state.position))

            # Getting the state data
            index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord = self.get_tip_coords()
            finger_state = list(index_tip_coord) + list(middle_tip_coord) + list(ring_tip_coord) + list(thumb_tip_coord)

            print("Current state data:\n Index-tip position: {}\n Middle-tip position: {}\n Ring-tip position: {}\n Thumb-tip position: {}\n".format(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord))

            cv2.imshow("Current state image", self.image)
            cv2.waitKey(1)

            # Transforming the image
            pil_image = PILImage.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            cropped_image = pil_image.crop((450, 220, 900, 920)) # Left, Top, Right, Bottom
            transformed_image_tensor = self.image_transform(cropped_image).unsqueeze(0)

            representation = self.encoder(transformed_image_tensor.float().to(self.device)).squeeze(2).squeeze(2).detach().cpu().numpy()[0]

            if self.debug is True:
                action, neighbor_idx, nn_state_image_path, trans_state_image_path = self.model.get_action_with_image(representation)
                action = list(action.reshape(12))

                l2_diff = np.linalg.norm(np.array(finger_state[:12]) - np.array(action))

                while l2_diff < self.threshold:
                    print("Obtaining new action!")
                    action, neighbor_idx, nn_state_image_path, trans_state_image_path = self.model.get_debug_action(neighbor_idx)
                    action = list(action.reshape(12))

                    l2_diff = np.linalg.norm(np.array(finger_state[:12]) - np.array(action))

                
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
                action = list(self.model.get_action(representation).reshape(12))
                # To write debug condition for k > 1

            print("Corresponding action: ", action)

            if self.abs is True:
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
    d = DexArmVINNDeploy(task = "rotate", debug = True, k = 1, device = "cuda", use_abs = True)
    d.deploy()