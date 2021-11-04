import os
import torch

def load_train_test_data(data_path = os.path.join(os.path.abspath(os.pardir), "data")):
    train_states, train_actions = load_state_actions(os.path.join(data_path, "train"))
    test_states, test_actions = load_state_actions(os.path.join(data_path, "test"))

    return train_states, train_actions, test_states, test_actions

def load_state_actions(data_path):
    # Getting the paths which contain the states and actions for each demo
    states_path, actions_path= os.path.join(data_path, "states"), os.path.join(data_path, "actions")

    # Getting the list of all demo states and actions
    states_demos_list, actions_demos_list = os.listdir(states_path), os.listdir(actions_path)

    assert len(states_demos_list) == len(actions_demos_list), "There are not equal number of state and action demo files!"

    # Sorting the state action pairs based on the demo numbers
    states_demos_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    actions_demos_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    states, actions = [], []

    for demo_idx in range(len(states_demos_list)):

        demo_states = torch.load(os.path.join(states_path, states_demos_list[demo_idx]))
        demo_actions = torch.load(os.path.join(actions_path, actions_demos_list[demo_idx]))

        assert len(demo_states) == len(demo_actions), "Number of states: {} from {} and number of actions: {} from are not equal.\n".format(demo_states.shape[0], states_demos_list[demo_idx], demo_actions.shape[0], actions_demos_list[demo_idx])

        states.append(demo_states)
        actions.append(demo_actions)

    # Converting the lists into a tensor
    states, actions = torch.cat(states), torch.cat(actions)

    return states, actions

def load_state_image_data(data_path):
    image_data_path = os.path.join(data_path, "images")
    demo_image_folders = os.listdir(image_data_path)
    demo_image_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    cumm_demos_image_count = []

    for demo_folder in demo_image_folders:
        demo_image_folder_path = os.path.join(image_data_path, demo_folder)
        demo_image_count = len(os.listdir(demo_image_folder_path))

        if len(cumm_demos_image_count) > 0:
            cumm_demos_image_count.append(cumm_demos_image_count[-1] + demo_image_count)
        else:
            cumm_demos_image_count.append(demo_image_count)

    return image_data_path, demo_image_folders, cumm_demos_image_count
            