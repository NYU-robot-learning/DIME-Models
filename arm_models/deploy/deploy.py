# Standard imports
import os
import numpy as np
import torch
import cv2
from PIL import Image as PILImage

# ROS imports
import rospy
from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError

# Controller imports
from ik_teleop.ik_core.allegro_ik import AllegroInvKDL
from move_dexarm import DexArmControl

# Model imports
from arm_models.imitation.networks import *
from arm_models.deploy.model_scripts import inn, visual_inn
from arm_models.utils.augmentation import augmentation_generator

# Miscellaneous imports
from copy import deepcopy as copy
import argparse

# Other torch imports
from torchvision import transforms as T

# ROS Topic to get data
ROBOT_IMAGE_TOPIC = "/cam_1/color/image_raw"
AR_MARKER_TOPIC = "/visualization_marker"
HAND_JOINT_STATE_TOPIC = "/allegroHand/joint_states"
ARM_JOINT_STATE_TOPIC = "/j2n6s300_driver/out/joint_state"

DATA_PATH = os.path.join(os.path.abspath(os.pardir), "data")
URDF_PATH = "/home/sridhar/dexterous_arm/ik_stuff/ik_teleop/urdf_template/allegro_right.urdf"

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str)
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-d', '--device', type=int)
parser.add_argument('--delta', type=int)
parser.add_argument('--run', type=int)

# Specific to INN or VINN
parser.add_argument('--k', type=int)
parser.add_argument('-p', '--target_priority', type=int)


class DexArmDeploy():
    def __init__(self, task, model, run, k = None, object_priority = 1, device = "cpu", threshold = 0.02):
        # Initializing ROS node for the dexterous arm
        print("Initializing ROS node.\n")
        rospy.init_node("deploy_node")

        self.task = task
        self.model_to_use = model
        self.threshold = threshold
        self.device = device
        self.k = k

        # Initializing the dexterous arm controller
        print("Initializing the controller for the dexterous arm.\n")
        self.arm = DexArmControl()

        # Positioning robot based on the task
        print("Positioning the dexterous arm to perform task: {}\n".format(task))
        # if task == "rotate" or task == "flip":
        #     self.arm.home_robot()
        # elif task == "spin":
            # self.arm.spin_pos_arm()

        # Initializing the Inverse Kinematics solver for the Allegro Hand
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = URDF_PATH)

        # Initializing the model for the specific task
        if self.model_to_use == "inn":
            print("Initializing INN for task {} with k = {}.\n".format(task, k))
            self.model = self._init_INN(task, k, object_priority, device)
        elif self.model_to_use == "vinn":
            print("Initializing VINN for task {} with k = {}.\n".format(task, k))
            self.model, self.learner = self._init_VINN(task, k, device, run)
        elif self.model_to_use == "mlp":
            print("Initializing a Simple MLP model for task {}.\n".format(task))
            self.model = self._init_MLP(task, device, run)
        elif self.model_to_use == "bc":
            print("Initializing a Behavior Cloning model for task {}.\n".format(task))
            self.model = self._init_BC(task, device, run)

        # Loading image transformer for image based models
        if self.model_to_use == "vinn" or self.model_to_use == "bc":
            if self.task == "rotate":
                self.image_transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((224, 224)),
                    T.Normalize(
                        mean = torch.tensor([0.4631, 0.4923, 0.5215]),
                        std = torch.tensor([0.2891, 0.2674, 0.2535])
                    )
                ])
            elif self.task == "flip":
                self.image_transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((224, 224)),
                    T.Normalize(
                        mean = torch.tensor([0.4534, 0.3770, 0.3885]),
                        std = torch.tensor([0.2512, 0.1881, 0.2599])
                    )
                ])
            elif self.task == "spin":
                self.image_transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((224, 224)),
                    T.Normalize(
                        mean = torch.tensor([0.4306, 0.3954, 0.3472]),
                        std = torch.tensor([0.2897, 0.2527, 0.2321])
                    )
                ])

        # Realtime data obtained through ROS
        self.allegro_joint_state = None
        self.kinova_joint_state = None 

        self.bridge = CvBridge()
        self.robot_image = None

        rospy.Subscriber(HAND_JOINT_STATE_TOPIC, JointState, self._callback_allegro_joint_state, queue_size = 1)
        rospy.Subscriber(ARM_JOINT_STATE_TOPIC, JointState, self._callback_kinova_joint_state, queue_size = 1)
        rospy.Subscriber(ROBOT_IMAGE_TOPIC, Image, self._callback_robot_image, queue_size=1)

        if self.task == "rotate" or self.task == "flip":
            rospy.Subscriber(AR_MARKER_TOPIC, Marker, self._callback_ar_marker_data, queue_size = 1)

        if self.task == "rotate":
            self.object_tracked_position = None
            self.hand_base_position = None
        elif self.task == "flip":
            self.object_tracked_position = None

    def _callback_allegro_joint_state(self, data):
        self.allegro_joint_state = data

    def _callback_kinova_joint_state(self, data):
        self.kinova_joint_state = data

    def _callback_robot_image(self, image):
        try:
            self.robot_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)

    def _callback_ar_marker_data(self, data):
        if data.id == 0 or data.id == 5:
            self.object_tracked_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        elif data.id == 8:
            self.hand_base_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])

    def _get_object_position(self):
        if self.task == "rotate":
            return self.object_tracked_position - self.hand_base_position
        elif self.task == "flip":
            return self.object_tracked_position

    def _init_INN(self, task, k, object_priority, device):
        # Getting the path to load the data
        if task == "rotate":
            folder = "cube_rotation"
        elif task == "flip":
            folder = "object_flipping"
        elif task == "spin":
            folder = "fidget_spinning"
        task_data_path = os.path.join(DATA_PATH, folder, "complete")

        # Initializing the model based on k to enable debug
        return inn.INNDeploy(
            k = k, 
            data_path = task_data_path, 
            load_image_data = True if k == 1 else False, 
            target_priority = object_priority, 
            device = device
        )

    def _init_VINN(self, task, k, device, run):
        # Loading the checkpoint based on the task
        chkpt_path = os.path.join(os.getcwd(), "checkpoints", "representation_byol - {} - lowest - train - v{}.pth".format(task, run))

        # Loading the representation encoder
        original_encoder_model = models.resnet50(pretrained = True)
        encoder = torch.nn.Sequential(*(list(original_encoder_model.children())[:-1]))
        encoder = encoder.to(device)

        learner = BYOL (
            encoder,
            image_size = 224 
        )
        learner.load_state_dict(torch.load(chkpt_path))
        learner.eval()

        # Loading the VINN model
        model =  visual_inn.VINNDeploy(
            k = k,
            load_image_data = True if k == 1 else False, 
            device = device
        )

        return model, learner

    def _init_MLP(self, task, device, run):
        # Loading the checkpoint based on the task
        chkpt_path = os.path.join(os.getcwd(), "checkpoints", "mlp - {} - lowest - train - v{}.pth".format(task, run))

        model = MLP().to(torch.device(device))
        model.load_state_dict(torch.load(chkpt_path))
        model.eval()
        return model

    def _init_BC(self, task, device, run):
        # Loading the checkpoint based on the task
        chkpt_path = os.path.join(os.getcwd(), "checkpoints", "bc - {} - lowest - train - v{}.pth".format(task, run))

        model = BehaviorCloning().to(torch.device(device))
        model.load_state_dict(torch.load(chkpt_path))
        model.eval()
        return model

    def _get_tip_coords(self):
        index_coord = self.allegro_ik.finger_forward_kinematics('index', list(self.allegro_joint_state.position)[:4])[0]
        middle_coord = self.allegro_ik.finger_forward_kinematics('middle', list(self.allegro_joint_state.position)[4:8])[0]
        ring_coord = self.allegro_ik.finger_forward_kinematics('ring', list(self.allegro_joint_state.position)[8:12])[0]
        thumb_coord = self.allegro_ik.finger_forward_kinematics('thumb', list(self.allegro_joint_state.position)[12:16])[0]

        return index_coord, middle_coord, ring_coord, thumb_coord

    def _update_joint_state(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord):
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

    def start(self, time_loop = False):
        print("\nDeploying model: {}\n".format(self.model_to_use))

        if time_loop is True:
            rate = rospy.Rate(2) # 2 sec sleep duration

        while True:
            if time_loop is False:
                next_step = input()

            # Checking if all the data streams are working
            if self.allegro_joint_state is None:
                print('No allegro state received!')
                continue

            if self.robot_image is None:
                print('No robot image received!')
                continue

            if self.task == "rotate":
                if self.object_tracked_position is None:
                    print("Object cannot be tracked!")
                    continue
                
                if self.hand_base_position is None:
                    print("Hand base position not found!")
                    continue

            if self.task == "flip":
                if self.object_tracked_position is None:
                    continue

            # Performing the step
            print("********************************************************\n             Starting new step \n********************************************************")

            # Displaying the current state data
            index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord = self._get_tip_coords()
            finger_state = list(index_tip_coord) + list(middle_tip_coord) + list(ring_tip_coord) + list(thumb_tip_coord)
            print("Current finger-tip positions:\n Index-tip: {}\n Middle-tip: {}\n Ring-tip: {}\n Thumb-tip: {}\n".format(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord))

            if self.k != 1:
                cv2.imshow("Current Robot State Image", self.robot_image)
                cv2.waitKey(1)

            if self.task == "rotate" or self.task == "flip":
                object_position = list(self._get_object_position())
                print("Object is located at position: {}".format(object_position))
            elif self.task == "spin":
                object_position = [0, 0, 0]

            if self.model_to_use == "vinn" or self.model_to_use == "bc":
                pil_image = PILImage.fromarray(cv2.cvtColor(self.robot_image, cv2.COLOR_BGR2RGB))
                if self.task == "rotate":
                    cropped_image = pil_image.crop((500, 160, 950, 600)) # Left, Top, Right, Bottom
                elif self.task == "flip": # TODO
                    cropped_image = pil_image.crop((220, 165, 460, 340)) # Left, Top, Right, Bottom
                elif self.task == "spin": # TODO
                    cropped_image = pil_image.crop((65, 80, 590, 480)) # Left, Top, Right, Bottom

                transformed_image_tensor = self.image_transform(cropped_image).unsqueeze(0)

            # Obtaining the actions through each model
            if self.model_to_use == "inn":
                if self.k == 1:
                    print("\nObtaining the action!\n")
                    action, neighbor_index, nn_state_image_path, trans_state_image_path, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, object_l2_diff = self.model.get_action_with_image(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position)
                    action = list(action.reshape(12))

                    l2_diff = np.linalg.norm(np.array(finger_state) - np.array(action))

                    if l2_diff >= self.threshold:
                        print("Distances:\n Index distance: {}\n Middle distance: {}\n Ring distance: {}\n Thumb distance: {}\n Cube distance: {}".format(index_l2_diff.item(), middle_l2_diff.item(), ring_l2_diff.item(), thumb_l2_diff.item(), object_l2_diff.item()))

                    while l2_diff < self.threshold:
                        print("Action distance is below threshold: {}".format(l2_diff))
                        action, neighbor_index, nn_state_image_path, trans_state_image_path = self.model.get_debug_action(neighbor_index)
                        # action, neighbor_index, nn_state_image_path, trans_state_image_path, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, object_l2_diff = self.model.get_action_with_image(action[:3], action[3:6], action[6:9], action[9:12], object_position)
                        action = list(action.reshape(12))
                        print("Primary neighbor index:", neighbor_index)

                        l2_diff = np.linalg.norm(np.array(finger_state) - np.array(action))

                    # Reading the Nearest Neighbor images 
                    nn_state_image = cv2.imread(nn_state_image_path)
                    trans_state_image = cv2.imread(trans_state_image_path)

                    # Combing the surrent state image and NN image and printing them
                    combined_images = np.concatenate((
                        cv2.resize(self.robot_image, (420, 240), interpolation = cv2.INTER_AREA), 
                        cv2.resize(nn_state_image, (420, 240), interpolation = cv2.INTER_AREA),
                        cv2.resize(trans_state_image, (420, 240), interpolation = cv2.INTER_AREA)
                    ), axis=1)

                    cv2.imshow("Current state and Nearest Neighbor Images", combined_images)
                    cv2.waitKey(1)

                else:
                    print("Obtaining the action!\n")
                    action = list(self.model.get_action(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position).reshape(12))
                    l2_diff = np.linalg.norm(np.array(finger_state) - np.array(action))

                    while l2_diff < self.threshold:
                        print("Action distance is below threshold: {}".format(l2_diff))
                        action = list(self.model.get_action(action[:3], action[3:6], action[6:9], action[9:12], object_position).reshape(12))
                        l2_diff = np.linalg.norm(np.array(finger_state) - np.array(action))
            
            elif self.model_to_use == "vinn":
                print("Image shape:", transformed_image_tensor.float().to(self.device).shape)
                representation = self.learner.net(transformed_image_tensor.float().to(self.device)).squeeze().detach().cpu().numpy()
                print("Representation shape:", representation.shape)
                if self.k == 1:
                    print("Obtaining the action!\n")
                    action, neighbor_idx, nn_state_image_path, trans_state_image_path = self.model.get_action_with_image(representation)
                    action = list(action.reshape(12))

                    l2_diff = np.linalg.norm(np.array(finger_state[:12]) - np.array(action))

                    while l2_diff < self.threshold:
                        print("Action distance is below threshold. Rolling out another action from the same trajectory!\n")
                        action, neighbor_idx, nn_state_image_path, trans_state_image_path = self.model.get_debug_action(neighbor_idx)
                        action = list(action.reshape(12))

                        l2_diff = np.linalg.norm(np.array(finger_state[:12]) - np.array(action))
                    
                    # Reading the Nearest Neighbor images 
                    nn_state_image = cv2.imread(nn_state_image_path)
                    trans_state_image = cv2.imread(trans_state_image_path)

                    # Combing the surrent state image and NN image and printing them
                    combined_images = np.concatenate((
                        cv2.resize(self.robot_image, (420, 240), interpolation = cv2.INTER_AREA), 
                        cv2.resize(nn_state_image, (420, 240), interpolation = cv2.INTER_AREA),
                        cv2.resize(trans_state_image, (420, 240), interpolation = cv2.INTER_AREA)
                    ), axis=1)

                    cv2.imshow("Current state and Nearest Neighbor Images", combined_images)
                    cv2.waitKey(1)

                else:
                    action = list(self.model.get_action(representation).reshape(12))

            elif self.model_to_use == "mlp":
                input_coordinates = torch.tensor(np.concatenate([index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position]))
                action = self.model(input_coordinates.float().to(self.device)).detach().cpu().numpy()[0]

            elif self.model_to_use == "bc":
                action = self.model(transformed_image_tensor.float().to(self.device)).detach().cpu().numpy()[0]

            # Updating the hand target coordinates
            updated_index_tip_coord = action[0:3]
            updated_middle_tip_coord = action[3:6]
            updated_ring_tip_coord = action[6:9]
            updated_thumb_tip_coord = action[9:12]

            print("Corresponding finger-tip action:\n Index-tip: {}\n Middle-tip: {}\n Ring-tip: {}\n Thumb-tip: {}\n".format(updated_index_tip_coord, updated_middle_tip_coord, updated_ring_tip_coord, updated_thumb_tip_coord))
            
            desired_joint_angles = self._update_joint_state(updated_index_tip_coord, updated_middle_tip_coord, updated_ring_tip_coord, updated_thumb_tip_coord)

            print("Moving arm to {}\n".format(desired_joint_angles))
            self.arm.move_hand(desired_joint_angles)

            if time_loop is True:
                rate.sleep()


if __name__ == '__main__':
    # Getting options
    options = parser.parse_args()

    d = DexArmDeploy(
        task = options.task, 
        model = options.model, 
        run = options.run,
        k = options.k, 
        object_priority = options.target_priority, 
        device = "cpu", 
        threshold = 0.01 * options.delta
    )
    
    d.start()