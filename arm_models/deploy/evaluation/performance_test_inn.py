from cProfile import label
import torch
import pickle

import matplotlib.pyplot as plt

from IPython import embed

from arm_models.imitation.non_parametric import INN
from arm_models.utils.load_data import *

DATA_PATH = "/home/sridhar/dexterous_arm/models/arm_models/data/cube_rotation/for_eval"

PERFORMANCE_METRICS_PATH = "/home/sridhar/dexterous_arm/models/arm_models/deploy/evaluation/losses/INN_losses"

def performance_test_inn(inn, k, test_states, test_actions):
    loss = 0
    
    # Testing the model
    for idx, test_state in enumerate(test_states):
        if k == 1:
            obtained_action, neighbor_index, index_l2_diff, middle_l2_diff, ring_l2_diff, thumb_l2_diff, cube_l2_diff = inn.find_optimum_action(test_state, k)
        else:
            obtained_action = inn.find_optimum_action(test_state, k)

        action_loss = torch.norm(obtained_action - test_actions[idx])

        loss += action_loss

    normalized_loss = loss / len(test_states)

    print("The testing loss is {}\n".format(normalized_loss))

    return normalized_loss

if __name__ == "__main__":
    k_losses = []

    # Getting the state and action pairs
    train_states, train_actions, test_states, test_actions = load_train_test_data(DATA_PATH)

    # Checking if the state action pairs are valid
    assert len(test_states) == len(test_actions), "The number of states is not equal to the number of actions!"
    print("Total number of train state and action pairs: {}".format(len(train_states)))
    print("Total number of test state and action pairs: {}".format(len(test_states)))
    
    print("\nTesting the model for object priority values from 1 to 50 and finger-tip priority values from 1 to 10.\n")

    plt.figure()

    # Computing the loss for 20 different k-values
    for finger_priority_idx in range(10):
        finger_losses = []

        for priority_idx in range(50):
            inn = INN(device="cuda", target_priority = priority_idx + 1, finger_priority =  finger_priority_idx + 1)

            # Loading data into the model
            inn.get_data(train_states, train_actions)

            print("Computing loss for priority = {}".format(priority_idx + 1))
            computed_loss = performance_test_inn(inn, 1, test_states, test_actions)

            finger_losses.append(computed_loss)

        print("The minimum observed loss was: {} for object priority: {}\n".format(min(finger_losses), finger_losses.index(min(finger_losses)) + 1))
        plt.plot([x+1 for x in range(50)], finger_losses, label="finger_priority = {}".format(finger_priority_idx + 1))
        plt.legend()
            


    # Saving the losses in a pickle file
    # loss_file = open(PERFORMANCE_METRICS_PATH, "ab")
    # pickle.dump(k_losses, loss_file)
    # loss_file.close()

    # Plotting the k-value based losses
    plt.xlabel("Object prioritized weights")

    plt.ylabel("Test loss")
    plt.show()
    # embed()
    