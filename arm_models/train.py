import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from arm_models.dataloaders.state_dataset import *
from arm_models.dataloaders.visual_dataset import *

from imitation.networks import *
from byol_pytorch import BYOL

import wandb
import argparse
from tqdm import tqdm

from utils.augmentation import augmentation_generator

CHKPT_PATH = os.path.join(os.getcwd(), "deploy", "checkpoints")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str)
parser.add_argument('-n', '--network', type=str)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-g', '--gpu_num', type=int)
parser.add_argument('-l', '--lr', type=float)
parser.add_argument('-i', '--img_size', type=int)
parser.add_argument('-d', '--train_data_path', type=str)
parser.add_argument('--full_data', action='store_true')
parser.add_argument('-v', '--val_data_path', type=str)
parser.add_argument('-f', '--test_data_path', type=str)
parser.add_argument('-a', '--augment_imgs', action='store_true')
parser.add_argument('-r', '--run', type=int)

if __name__ == '__main__':
    # Getting the parser options
    options = parser.parse_args()

    # Selecting the torch device
    device = torch.device("cuda:{}".format(options.gpu_num))
    print("Using GPU: {} for training. \n".format(options.gpu_num))

    # Load the dataset and creating dataloader
    print("Loading dataset(s) and creating dataloader(s).\n")
    if options.task == "rotate":
        if options.img_size is None:
            train_dataset = CubeRotationDataset(options.train_data_path)
            if options.val_data_path is not None:
                val_dataset = CubeRotationDataset(options.val_data_path)
            if options.test_data_path is not None:
                test_dataset = CubeRotationDataset(options.test_data_path)
        else:
            if options.network == "bc":
                if options.full_data is True:
                    print("Using the entire dataset!\n")
                    train_dataset = CubeRotationVisualDataset(type = None, image_size = options.img_size, data_path = options.train_data_path)
                else:
                    train_dataset = CubeRotationVisualDataset(type = "train", image_size = options.img_size, data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = CubeRotationVisualDataset(type = "val", image_size = options.img_size, data_path = options.val_data_path)
                if options.test_data_path is not None:
                    test_dataset = CubeRotationVisualDataset(type = "test", image_size = options.img_size, data_path = options.test_data_path)
            elif options.network == "representation_byol":
                train_dataset = RepresentationVisualDataset(task = "cube_rotation", data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = RepresentationVisualDataset(task = "cube_rotation", data_path = options.train_data_path)
                if options.test_data_path is not None:
                    test_dataset = RepresentationVisualDataset(task = "cube_rotation", data_path = options.train_data_path)

    elif options.task == "flip":
        if options.img_size is None:
            train_dataset = ObjectFlippingDataset(options.train_data_path)
            if options.val_data_path is not None:
                val_dataset = ObjectFlippingDataset(options.val_data_path)
            if options.test_data_path is not None:
                test_dataset = ObjectFlippingDataset(options.test_data_path)
        else:
            if options.network == "bc":
                if options.full_data is True:
                    print("Using the entire dataset!\n")
                    train_dataset = ObjectFlippingVisualDataset(type = None, image_size = options.img_size, data_path = options.train_data_path)
                else:
                    train_dataset = ObjectFlippingVisualDataset(type = "train", image_size = options.img_size, data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = ObjectFlippingVisualDataset(type = "val", image_size = options.img_size, data_path = options.val_data_path)
                if options.test_data_path is not None:
                    test_dataset = ObjectFlippingVisualDataset(type = "test", image_size = options.img_size, data_path = options.test_data_path)
            elif options.network == "representation_byol":
                train_dataset = RepresentationVisualDataset(task = "object_flipping", data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = RepresentationVisualDataset(task = "object_flipping", data_path = options.train_data_path)
                if options.test_data_path is not None:
                    test_dataset = RepresentationVisualDataset(task = "object_flipping", data_path = options.train_data_path)

    elif options.task == "spin":
        if options.img_size is None:
            train_dataset = FidgetSpinningDataset(options.train_data_path) 
            if options.val_data_path is not None:
                val_dataset = FidgetSpinningDataset(options.val_data_path)
            if options.test_data_path is not None:
                test_dataset = FidgetSpinningDataset(options.test_data_path)
        else:
            if options.network == "bc":
                if options.full_data is True:
                    print("Using the entire dataset!\n")
                    train_dataset = FidgetSpinningVisualDataset(type = None, image_size = options.img_size, data_path = options.train_data_path)
                else:
                    train_dataset = FidgetSpinningVisualDataset(type = "train", image_size = options.img_size, data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = FidgetSpinningVisualDataset(type = "val", image_size = options.img_size, data_path = options.val_data_path)
                if options.test_data_path is not None:
                    test_dataset = FidgetSpinningVisualDataset(type = "test", image_size = options.img_size, data_path = options.test_data_path)
            elif options.network == "representation_byol":
                train_dataset = RepresentationVisualDataset(task = "fidget_spinning", data_path = options.train_data_path)
                if options.val_data_path is not None:
                    val_dataset = RepresentationVisualDataset(task = "fidget_spinning", data_path = options.train_data_path)
                if options.test_data_path is not None:
                    test_dataset = RepresentationVisualDataset(task = "fidget_spinning", data_path = options.train_data_path)

    train_dataloader = DataLoader(train_dataset, options.batch_size, shuffle = True, pin_memory = True, num_workers = 24)

    if options.val_data_path is not None:
        val_dataloader = DataLoader(val_dataset, options.batch_size, shuffle = True, pin_memory = True, num_workers = 24)

    if options.test_data_path is not None:
        test_dataloader = DataLoader(test_dataset, options.batch_size, shuffle = True, pin_memory = True, num_workers = 24)

    # Initializing the model based on the model argument
    print("Loading the model({}) for the task: {}\n".format(options.network, options.task))
    if options.network == "mlp":
        model = MLP().to(device)
    elif options.network == "bc":
        model = BehaviorCloning().to(device)
    elif options.network == "representation_byol":
        original_encoder_model = models.resnet50(pretrained = True)
        encoder = torch.nn.Sequential(*(list(original_encoder_model.children())[:-1])).to(device)

        if options.augment_imgs is True:
            augment_custom = augmentation_generator(options.task)
            model = BYOL(
                encoder,
                image_size = options.img_size,
                augment_fn = augment_custom
            )
        else:
            model = BYOL(
                encoder,
                image_size = options.img_size
            )            

    # Initialize WandB logging
    wandb.init(project = "{} - {}".format(options.task, options.network))

    # Initializing optimizer and other parameters
    optimizer = torch.optim.Adam(model.parameters(), lr = options.lr)
    
    if options.test_data_path is not None:
        low_test_loss = np.inf
    elif options.val_data_path is not None:
        low_val_loss = np.inf
    else:
        low_train_loss = np.inf

    # Initializing loss
    if options.network != "representation_byol":
        loss_fn = nn.MSELoss()

    # Training loop
    print("Starting training procedure!\n")
    for epoch in range(options.epochs):
        # Training part
        epoch_train_loss = 0

        for input_data, actions in tqdm(train_dataloader):
            optimizer.zero_grad()

            if options.network == "representation_byol":
                loss = model(input_data.float().to(device))
            else:
                predicted_actions = model(input_data.float().to(device))
                loss = loss_fn(predicted_actions, actions.float().to(device))
            
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * input_data.shape[0]

        print("Train loss: {}\n".format(epoch_train_loss / len(train_dataset)))
        wandb.log({'train loss': epoch_train_loss / len(train_dataset)})

        # Validation part
        if options.val_data_path is not None:
            epoch_val_loss = 0

            for input_data, actions in tqdm(val_dataloader):
                if options.network == "representation_byol":
                    loss = model(input_data.float().to(device))
                else:
                    predicted_actions = model(input_data.float().to(device))
                    loss = loss_fn(predicted_actions, actions.float().to(device))

                epoch_val_loss += loss.item() * input_data.shape[0]

            print("Validation loss: {}\n".format(epoch_val_loss / len(val_dataset)))
            wandb.log({'validation loss': epoch_val_loss / len(val_dataset)})

        # Testing part
        if options.test_data_path is not None:
            epoch_test_loss = 0

            for input_data, actions in tqdm(test_dataloader):
                if options.network == "representation_byol":
                    loss = model(input_data.float().to(device))
                else:
                    predicted_actions = model(input_data.float().to(device))
                    loss = loss_fn(predicted_actions, actions.float().to(device))

                epoch_test_loss += loss.item() * input_data.shape[0]

            print("Test loss: {}\n".format(epoch_test_loss / len(test_dataset)))
            wandb.log({'test loss': epoch_test_loss / len(test_dataset)})



        # Saving checkpoints based on the lowest stats
        if options.network == "vinn":
            low_train_loss = epoch_train_loss / len(train_dataset)
            epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - lowest - train - v{}.pth".format(options.network, options.task, options.run))
            print("\nLower train loss encountered! Saving checkpoint {}\n".format(epoch_checkpt_path))
            torch.save(encoder.state_dict(), epoch_checkpt_path)
        else:
            if options.test_data_path is not None:
                if epoch_test_loss / len(test_dataset) < low_test_loss:
                    low_test_loss = epoch_test_loss / len(test_dataset)
                    epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - lowest - test - v{}.pth".format(options.network, options.task, options.run))
                    print("\nLower test loss encountered! Saving checkpoint {}\n".format(epoch_checkpt_path))
                    torch.save(model.state_dict(), epoch_checkpt_path)
            elif options.val_data_path is not None:
                if epoch_val_loss / len(val_dataset) < low_val_loss:
                    low_val_loss = epoch_val_loss / len(val_dataset)
                    epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - lowest - val - v{}.pth".format(options.network, options.task, options.run))
                    print("\nLower validation loss encountered! Saving checkpoint {}\n".format(epoch_checkpt_path))
                    torch.save(model.state_dict(), epoch_checkpt_path)
            else:
                if epoch_train_loss / len(train_dataset) < low_train_loss:
                    low_train_loss = epoch_train_loss / len(train_dataset)
                    epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - lowest - train - v{}.pth".format(options.network, options.task, options.run))
                    print("\nLower train loss encountered! Saving checkpoint {}\n".format(epoch_checkpt_path))
                    torch.save(model.state_dict(), epoch_checkpt_path)

    # Saving final checkpoint
    if options.task == "vinn":
        final_epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - final - v{}.pth".format(options.network, options.task, options.run))
        print("\nSaving final checkpoint {}\n".format(final_epoch_checkpt_path))
        torch.save(encoder.state_dict(), final_epoch_checkpt_path)
    else:
        final_epoch_checkpt_path = os.path.join(CHKPT_PATH, "{} - {} - final - v{}.pth".format(options.network, options.task, options.run))
        print("\nSaving final checkpoint {}\n".format(final_epoch_checkpt_path))
        torch.save(model.state_dict(), final_epoch_checkpt_path)