import torch
from torch.utils.data import DataLoader
from torchvision import models

from tqdm import tqdm

from arm_models.dataloaders.visual_dataset import *

from byol_pytorch import BYOL

REP_TENSOR_PATH = '/home/sridhar/dexterous_arm/models/arm_models/data/fidget_spinning/complete/representations/representations.pth'
REP_MODEL_CHKPT_PATH = '/home/sridhar/dexterous_arm/models/arm_models/deploy/checkpoints/representation_byol - spin - lowest - train - v1.pth'

def extract_representations(device, CHKPT_PATH):
    # Loading the dataset and creating the dataloader
    # dataset = CubeRotationVisualDataset()
    # dataset = ObjectFlippingVisualDataset()
    dataset = FidgetSpinningVisualDataset()

    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 24)

    # Loading the model
    original_encoder_model = models.resnet50(pretrained = True)
    encoder = torch.nn.Sequential(*(list(original_encoder_model.children())[:-1]))
    encoder = encoder.to(device)

    learner = BYOL (
        encoder,
        image_size = 224 
    )
    learner.load_state_dict(torch.load(CHKPT_PATH))
    learner.eval()

    representations = []

    # Extracting the representations
    for image, action in tqdm(dataloader):
        representation = learner.net(image.float().to(device)).squeeze()
        representations.append(representation.detach().cpu())

    representation_tensor = torch.stack(representations)
    print("Final representation tensor shape:", representation_tensor.squeeze().shape)

    return representation_tensor.squeeze(1)

def store_representations(representations, DATA_PATH):
    torch.save(representations, DATA_PATH)

if __name__ == '__main__':
    # Selecting the GPU to be used
    gpu_number = int(input('Enter the GPU number: '))
    device = torch.device('cuda:{}'.format(gpu_number))

    print('Using GPU: {} for extracting the representations. \n'.format(torch.cuda.get_device_name(gpu_number)))

    representation_tensor = extract_representations(device, CHKPT_PATH = REP_MODEL_CHKPT_PATH)
    store_representations(representation_tensor, REP_TENSOR_PATH)