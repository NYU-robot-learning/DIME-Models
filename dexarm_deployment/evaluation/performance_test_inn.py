import torch
import pickle

import matplotlib.pyplot as plt

from IPython import embed

from Imitation_NN.non_parametric_imitation import ImitationNearestNeighbors
from utils.dataloader import *

DATA_PATH = "/home/sridhar/dexterous_arm/models/data/for_eval"

PERFORMANCE_METRICS_PATH = "/home/sridhar/dexterous_arm/models/dexarm_deployment/evaluation/losses/INN_losses"

def performance_test_inn(data_path, k):
    inn = ImitationNearestNeighbors(device="cuda")
    
    # Getting the state and action pairs
    train_states, train_actions, test_states, test_actions = load_train_test_data(DATA_PATH)

    # Loading data into the model
    inn.get_data(train_states, train_actions)

    # Checking if the state action pairs are valid
    assert len(test_states) == len(test_actions), "The number of states is not equal to the number of actions!"
    print("Total number of test state and action pairs: {}".format(len(test_states)))

    print("Testing the model...")

    loss = 0
    
    # Testing the model
    for idx, test_state in enumerate(test_states):
        if k == 1:
            obtained_action, neighbor_index = inn.find_optimum_action(test_state, k)
        else:
            obtained_action = inn.find_optimum_action(test_state, k)

        action_loss = torch.norm(obtained_action - test_actions[idx])

        loss += action_loss

    normalized_loss = loss / len(test_states)

    print("The testing loss is {}\n".format(normalized_loss))

    return normalized_loss

if __name__ == "__main__":
    k_losses = []
    
    # Computing the loss for 20 different k-values
    for k in range(20):
        print("Computing loss for k = {}".format(k + 1))
        computed_loss = performance_test_inn(DATA_PATH, k + 1)

        k_losses.append(computed_loss)

    # Saving the losses in a pickle file
    loss_file = open(PERFORMANCE_METRICS_PATH, "ab")
    pickle.dump(k_losses, loss_file)
    loss_file.close()

    # Plotting the k-value based losses
    plt.figure()
    plt.plot([x+1 for x in range(20)], k_losses)
    plt.show()

    # embed()
    