import torch

class ImitationNearestNeighbors():
    def __init__(self, device="cpu", cube_priority = 1):
        # Setting the device
        self.device = torch.device(device)

        # Initializing the states and action pairs to None
        self.states = None
        self.actions = None

        self.cube_priority = cube_priority

    def get_data(self, states, actions):
        self.states = states.to(self.device)
        self.actions = actions.to(self.device)

    def getNearestNeighbors(self, input_state, k):
        # Comparing the dataset shape and state
        assert torch.tensor(input_state).shape == self.states[0].shape, "There is a data shape mismatch: \n Shape of loaded dataset: {} \n Shape of current state: {}".format(input_state.shape, self.states[0].shape)

        # Getting the k-Nearest Neighbor actions
        state_diff = self.states - torch.tensor(input_state).to(self.device)

        index_l2_diff = torch.norm(state_diff[:, :3], dim=1)
        middle_l2_diff = torch.norm(state_diff[:, 3:6], dim=1)
        ring_l2_diff = torch.norm(state_diff[:, 6:9], dim=1)
        thumb_l2_diff = torch.norm(state_diff[:, 9:12], dim=1)
        cube_l2_diff = torch.norm(state_diff[:, 12:], dim=1)

        l2_diff = index_l2_diff + middle_l2_diff + ring_l2_diff + thumb_l2_diff + (cube_l2_diff * self.cube_priority)


        sorted_idxs = torch.argsort(l2_diff)[:k]

        if k == 1: 
            k_nn_actions = self.actions[torch.argsort(l2_diff)[0]]
            return k_nn_actions, torch.argsort(l2_diff)[0].cpu().detach(), index_l2_diff[sorted_idxs], middle_l2_diff[sorted_idxs], ring_l2_diff[sorted_idxs], thumb_l2_diff[sorted_idxs], cube_l2_diff[sorted_idxs]
        else:
            return self.actions[torch.argsort(l2_diff)[:k]]


    def find_optimum_action(self, input_state, k):
        # Getting the k-Nearest Neighbor actions for the input state
        if k == 1:
            k_nn_action, neighbor_idx, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, cube_l2_diff = self.getNearestNeighbors(input_state, k)
            return k_nn_action.cpu().detach(), neighbor_idx, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, cube_l2_diff
        else:
            k_nn_actions = self.getNearestNeighbors(input_state, k)

            # Getting the mean value from the set of nearest neighbor states
            mean_action = torch.mean(k_nn_actions, 0).cpu().detach()
            return mean_action