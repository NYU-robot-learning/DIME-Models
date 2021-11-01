import torch

class ImitationNearestNeighbors():
    def __init__(self, device="cpu"):
        # Setting the device
        self.device = torch.device(device)

        # Initializing the states and action pairs to None
        self.states = None
        self.actions = None

    def get_data(self, states, actions):
        self.states = states.to(self.device)
        self.actions = actions.to(self.device)

    def getNearestNeighbors(self, input_state, k):
        # Comparing the dataset shape and state
        assert input_state.shape == self.states[0].shape, "There is a data shape mismatch: \n Shape of loaded dataset: {} \n Shape of current state: {}".format(input_state.shape, self.states[0].shape)

        # Getting the k-Nearest Neighbor actions
        state_diff = self.states - torch.tensor(input_state).to(self.device)
        l2_diff = torch.norm(state_diff, dim=1)
        k_nn_actions = self.actions[torch.argsort(l2_diff)[:k]]

        return k_nn_actions

    def find_optimum_action(self, input_state, k):
        # Getting the k-Nearest Neighbor actions for the input state
        k_nn_actions = self.getNearestNeighbors(input_state, k)

        # Getting the mean value from the set of nearest neighbor states
        mean_action = torch.mean(k_nn_actions, 0).cpu().detach()

        return mean_action