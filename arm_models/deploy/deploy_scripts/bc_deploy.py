# Standard imports
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
from arm_models.imitation_models.Behavior_Cloning.model import BehaviorCloning

# Other imports
from copy import deepcopy as copy
from torchvision import transforms as T

# ROS Topics to get the AR Marker data
IMAGE_TOPIC = "/cam_1/color/image_raw"
JOINT_STATE_TOPIC = "/allegroHand/joint_states"

MODEL_CHKPT_PATH = '/home/sridhar/dexterous_arm/models/arm_models/dexarm_deployment/model_checkpoints/behavior_cloning.pth'

class DexArmMLPDeploy():
    def __init__(self, device = 'cpu'):
        # Ignoring scientific notations
        torch.set_printoptions(precision=None)

        # Initializing ROS deployment node
        try:
            rospy.init_node("model_deploy")
        except:
            pass

        # Initializing arm controller
        self.arm = DexArmControl()

        # Initializing Allegro Ik Library
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/ik_teleop/urdf_template/allegro_right.urdf")

        # Loading the model and assigning device
        self.device = torch.device(device)
        self.model = BehaviorCloning()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(MODEL_CHKPT_PATH))
        self.model.eval()

        # Moving the dexterous arm to home position
        self.arm.home_robot()

        # Initializing topic data
        self.image = None
        self.allegro_joint_state = None

        # Initalizing image transformer
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
            # Waiting for key
            next_step = input()

            # Setting the break condition
            if next_step == "q":
                break

            print("Getting state data...\n")

            print("Current joint angles: {}\n".format(self.allegro_joint_state.position))

            # Getting the state data
            index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord = self.get_tip_coords()

            print("Current state data:\n Index-tip position: {}\n Middle-tip position: {}\n Ring-tip position: {}\n Thumb-tip position: {}\n".format(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord))

            cv2.imshow("Current state", self.image)
            cv2.waitKey(1)

            # Transforming the image before passing it into the model
            pil_image = PILImage.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            cropped_image = pil_image.crop((450, 220, 900, 920)) # Left, Top, Right, Bottom
            transformed_image_tensor = self.image_transform(cropped_image).unsqueeze(0)

            action = self.model(transformed_image_tensor.float().to(self.device)).detach().cpu().numpy()[0]

            print("Corresponding action: ", action)

            # Updating the hand target coordinates
            updated_index_tip_coord = action[0:3]
            updated_middle_tip_coord = action[3:6]
            updated_ring_tip_coord = action[6:9]
            updated_thumb_tip_coord = action[9:12]

            print("updated index tip coord:", updated_index_tip_coord)
            
            desired_joint_angles = self.update_joint_state(updated_index_tip_coord, updated_middle_tip_coord, updated_ring_tip_coord, updated_thumb_tip_coord)

            print("Moving arm to {}\n".format(desired_joint_angles))
            self.arm.move_hand(desired_joint_angles)

if __name__ == '__main__':
    d = DexArmMLPDeploy(device = 'cpu')
    d.deploy()