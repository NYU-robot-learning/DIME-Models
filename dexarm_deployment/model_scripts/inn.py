import torch

from Imitation_NN.non_parametric_imitation import ImitationNearestNeighbors
from utils.dataloader import load_state_actions

DATA_PATH = "/home/sridhar/dexterous_arm/models/data"

class INNDeploy():
    def __init__(self, data_path = DATA_PATH, k = 1, device = "cpu"):
        self.model = ImitationNearestNeighbors(device)
        self.k = k

        states, actions = load_state_actions(data_path)
        self.model.get_data(states, actions)

    def get_action(self, thumb_tip_coord, ring_tip_coord, cube_pos):
        state = list(thumb_tip_coord) + list(ring_tip_coord) + list(cube_pos)
        calculated_action = self.model.find_optimum_action(state, self.k)
        return calculated_action.cpu().detach().numpy()