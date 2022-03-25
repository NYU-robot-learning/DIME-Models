import os

from arm_models.imitation.non_parametric import VINN
from arm_models.utils.load_data import load_representation_actions, load_state_image_data

DATA_PATH = "/home/sridhar/dexterous_arm/models/arm_models/data/fidget_spinning/complete"

class VINNDeploy():
    def __init__(self, data_path = DATA_PATH, k = 1, load_image_data = True, device = "cpu"):
        self.k = k
        self.image_data_path = None

        self.representations, self.actions = load_representation_actions(data_path)

        self.model = VINN(device)
        self.model.get_data(self.representations, self.actions)

        if load_image_data is True:
            print(data_path)
            self.image_data_path, self.demo_image_folders, self.cumm_demos_image_count = load_state_image_data(data_path)

    def get_action(self, representation):
        if self.k == 1:
            action, neighbor_idx =  self.model.find_optimum_action(representation, self.k)
            return action.detach().cpu().numpy()
        else:
            return self.model.find_optimum_action(representation, self.k).detach().cpu().numpy()

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

    def get_action_with_image(self, representation):
        if self.k == 1 and self.image_data_path is not None:
            # state = list(index_tip_coord) + list(middle_tip_coord) + list(ring_tip_coord) + list(thumb_tip_coord) + list(cube_pos)
            calculated_action, neighbor_index = self.model.find_optimum_action(representation, self.k)

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

            return calculated_action.cpu().detach().numpy(), neighbor_index, nn_state_image_path, trans_state_image_path
        else:
            return self.get_action(representation)