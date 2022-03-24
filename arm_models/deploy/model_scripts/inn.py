import os
import torch

from arm_models.imitation.non_parametric import INN
from arm_models.utils.load_data import load_state_actions, load_state_image_data

class INNDeploy():
    def __init__(self, data_path, target_priority = 1, k = 1, load_image_data = False, device = "cpu"):
        self.k = k
        self.image_data_path = None

        self.states, self.actions = load_state_actions(data_path)

        self.model = INN(device, target_priority = target_priority)
        self.model.get_data(self.states, self.actions)

        if load_image_data is True:
            print("Loading data from path:", data_path)
            self.image_data_path, self.demo_image_folders, self.cumm_demos_image_count = load_state_image_data(data_path)

    def get_action(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position):
        state = list(index_tip_coord) + list(middle_tip_coord) + list(ring_tip_coord) + list(thumb_tip_coord) + list(object_position)
        if self.k == 1:
            action, neighbor_idx, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, object_l2_diff =  self.model.find_optimum_action(state, self.k)
            return action.cpu().detach().numpy()
        else:
            return self.model.find_optimum_action(state, self.k).cpu().detach().numpy()

    def get_action_with_image(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position):
        if self.k == 1 and self.image_data_path is not None:
            state = list(index_tip_coord) + list(middle_tip_coord) + list(ring_tip_coord) + list(thumb_tip_coord) + list(object_position)
            calculated_action, neighbor_index, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, object_l2_diff = self.model.find_optimum_action(state, self.k)

            if neighbor_index < self.cumm_demos_image_count[0]:
                nn_image_num = neighbor_index
                nn_trans_image_num = nn_image_num + 1
                demo_num = 0
            else:
                for idx, cumm_demo_images in enumerate(self.cumm_demos_image_count):
                    if neighbor_index + 1 > cumm_demo_images:
                        # Obtaining the demo number
                        demo_num = idx + 1

                        # Getting the corresponding image number
                        nn_image_num = neighbor_index - cumm_demo_images
                        nn_trans_image_num = nn_image_num + 1

            nn_state_image_path = os.path.join(self.image_data_path, self.demo_image_folders[demo_num], "state_{}.jpg".format(nn_image_num.item()))
            trans_state_image_path = os.path.join(self.image_data_path, self.demo_image_folders[demo_num], "state_{}.jpg".format(nn_trans_image_num.item()))

            return calculated_action.cpu().detach().numpy(), neighbor_index, nn_state_image_path, trans_state_image_path, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, object_l2_diff
        else:
            self.get_action(index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord, object_position)

    def get_debug_action(self, neighbor_index):
        if neighbor_index < self.cumm_demos_image_count[0]:
            nn_image_num = neighbor_index
            nn_trans_image_num = nn_image_num + 1
            demo_num = 0
        else:
            for idx, cumm_demo_images in enumerate(self.cumm_demos_image_count):
                if neighbor_index + 1 > cumm_demo_images:
                    # Obtaining the demo number
                    demo_num = idx + 1

                    # Getting the corresponding image number
                    nn_image_num = neighbor_index - cumm_demo_images
                    nn_trans_image_num = nn_image_num + 1

        nn_state_image_path = os.path.join(self.image_data_path, self.demo_image_folders[demo_num], "state_{}.jpg".format(nn_image_num.item()))
        trans_state_image_path = os.path.join(self.image_data_path, self.demo_image_folders[demo_num], "state_{}.jpg".format(nn_trans_image_num.item()))

        return self.actions[neighbor_index + 1], neighbor_index + 1, nn_state_image_path, trans_state_image_path