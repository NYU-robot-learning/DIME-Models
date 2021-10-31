import os
import torch

from utils.dataloader import load_train_test_data

class ImitationNearestNeighbors():
    def __init__(self,
        device="cpu",
        data_path = os.path.join(os.getcwd(), "data")):

        # Setting the device
        self.device = torch.device(device)

        # Loading dataset and transfering it to device         
        train_states, train_actions, self.test_states, self.test_actions = load_train_test_data(data_path)

        self.train_states, self.train_actions = train_states.to(device), train_actions.to(device)

    def getNearestNeighbors(self, input_state, k):

        # Comparing the dataset shape and state
        assert input_state.shape != self.train_state[0].shape, "There is a data shape mismatch: \n Shape of loaded dataset: {} \n Shape of current state: {}"

        # Getting the k-Nearest Neighbor actions
        state_diff = self.train_states - torch.tensor(input_state).to(self.device)
        l2_diff = torch.norm(state_diff, dim=1)
        nn_actions = self.train_actions[torch.argsort(l2_diff)]
        k_nn_actions = nn_actions[:k+1]

        return k_nn_actions

    def find_optimum_action(self, input_state, k):
        # Getting the k-Nearest Neighbor actions for the input state
        k_nn_actions = self.getNearestNeighbors(input_state, k)

        # Getting the mean value from the set of nearest neighbor states
        mean_action = torch.mean(k_nn_actions).cpu().detach()

        return mean_action