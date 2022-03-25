from argparse import Action
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from tqdm import tqdm

class RepresentationVisualDataset(Dataset):
    def __init__(self, task, image_size = 224, data_path = '/home/sridhar/dexterous_arm/demonstrations/image_data'):
        # Loading the images
        images_dir= os.path.join(data_path, task)
        
        # Sorting all the demo paths
        image_path_list = os.listdir(images_dir)
        image_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.image_tensors = []

        # Transforming the images
        if task == "cube_rotation":
            mean = torch.tensor([0.4640, 0.4933, 0.5223])
            std = torch.tensor([0.2890, 0.2671, 0.2530])
        elif task == "object_flipping":
            mean = torch.tensor([0.4815, 0.5450, 0.5696])
            std = torch.tensor([0.2291, 0.2268, 0.2248]) 
        elif task == "fidget_spinning":
            mean = torch.tensor([0.4306, 0.3954, 0.3472])
            std = torch.tensor([0.2897, 0.2527, 0.2321]) 

        self.image_preprocessor = T.Compose([
            T.ToTensor(),
            T.Resize((image_size, image_size)),
            T.Normalize(
                mean = mean, 
                std = std  
            )
        ])

        # Loading all the images in the images vector
        print("Loading all the images: \n")
        for demo_images_path in tqdm(image_path_list):
            demo_image_folder_path = os.path.join(images_dir, demo_images_path)

            # Sort the demo images list
            demo_images_list = os.listdir(demo_image_folder_path)

            demo_images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            # Read each image and append them in the images array
            for idx in range(len(demo_images_list) - 1):
                try:
                    image = Image.open(os.path.join(demo_image_folder_path, demo_images_list[idx]))
                    if task == "cube_rotation":
                        image = image.crop((500, 160, 950, 600)) # Left, Top, Right, Bottom
                    elif task == "object_flipping":
                        image = image.crop((220, 165, 460, 340)) # Left, Top, Right, Bottom
                    elif task == "fidget_spinning":
                        image = image.crop((65, 80, 590, 480)) # Left, Top, Right, Bottom
                    
                    image_tensor = self.image_preprocessor(image)
                    self.image_tensors.append(image_tensor.detach())
                    image.close()
                except:
                    print('Image cannot be read!')
                    continue
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return (self.image_tensors[idx], torch.zeros(12))

class CubeRotationVisualDataset(Dataset):
    def __init__(self, type = None, image_size = 224, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/cube_rotation'):
        if type == "train":
            self.data_path = os.path.join(data_path, "for_eval", "train")
        elif type == "val":
            self.data_path = os.path.join(data_path, "for_eval", "validation")
        elif type == "test":
            self.data_path = os.path.join(data_path, "for_eval", "test")
        else:
            self.data_path = os.path.join(data_path, "complete")

        # Loading all the image and action folder paths
        images_dir, actions_dir = os.path.join(self.data_path, 'images'), os.path.join(self.data_path, 'actions')
        
        # Sorting all the demo paths
        image_path_list, action_path_list = os.listdir(images_dir), os.listdir(actions_dir)

        image_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        action_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.image_tensors, self.action_tensors = [], []

        # Transforming the images
        if type == "train":
            mean = torch.tensor([0.4631, 0.4917, 0.5201])
            std = torch.tensor([0.2885, 0.2666, 0.2526])
        elif type == "val":
            mean = torch.tensor([0.3557, 0.4138, 0.4550])
            std = torch.tensor([0.9030, 0.8565, 0.8089])
        elif type == "load":
            mean = torch.tensor([0, 0, 0])
            std = torch.tensor([1, 1, 1])
        else:
            mean = torch.tensor([0.4631, 0.4923, 0.5215])
            std = torch.tensor([0.2891, 0.2674, 0.2535])

        self.image_preprocessor = T.Compose([
            T.ToTensor(),
            T.Resize((image_size, image_size)),
            T.Normalize(
                mean = mean, 
                std = std 
            )
        ])

        # Loading all the images in the images vector
        print("Loading all the images: \n")
        for demo_images_path in tqdm(image_path_list):
            demo_image_folder_path = os.path.join(images_dir, demo_images_path)

            # Sort the demo images list
            demo_images_list = os.listdir(demo_image_folder_path)
            demo_images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            # print("Reading images from {}".format(demo_image_folder_path))

            # Read each image and append them in the images array
            for idx in range(len(demo_images_list) - 1):
                try:
                    image = Image.open(os.path.join(demo_image_folder_path, demo_images_list[idx]))
                    image = image.crop((500, 160, 950, 600)) # Left, Top, Right, Bottom
                    image_tensor = self.image_preprocessor(image)
                    self.image_tensors.append(image_tensor.detach())
                    image.close()
                except:
                    print('Image cannot be read!')
                    continue

        # Loading all the action vectors
        print("\nLoading all the actions: \n")
        for demo_action_path in tqdm(action_path_list):
            demo_actions = torch.load(os.path.join(actions_dir, demo_action_path))
            self.action_tensors.append(demo_actions)
        
        self.action_tensors = torch.cat(self.action_tensors, dim = 0)
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return ((self.image_tensors[idx], self.action_tensors[idx]))

class ObjectFlippingVisualDataset(Dataset):
    def __init__(self, type = None, image_size = 224, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/object_flipping'):
        if type == "train":
            self.data_path = os.path.join(data_path, "for_eval", "train")
        elif type == "val":
            self.data_path = os.path.join(data_path, "for_eval", "validation")
        elif type == "test":
            self.data_path = os.path.join(data_path, "for_eval", "test")
        else:
            self.data_path = os.path.join(data_path, "complete")

        # Loading all the image and action folder paths
        images_dir, actions_dir = os.path.join(self.data_path, 'images'), os.path.join(self.data_path, 'actions')
        
        # Sorting all the demo paths
        image_path_list, action_path_list = os.listdir(images_dir), os.listdir(actions_dir)

        image_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        action_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.image_tensors, self.action_tensors = [], []

        # Transforming the images
        if type == "train":
            mean = torch.tensor([0.4631, 0.4917, 0.5201]) # Recalculate
            std = torch.tensor([0.2885, 0.2666, 0.2526]) # Recalculate
        elif type == "val":
            mean = torch.tensor([0.3557, 0.4138, 0.4550]) # Recalculate
            std = torch.tensor([0.9030, 0.8565, 0.8089]) # Recalculate
        elif type == "load":
            mean = torch.tensor([0, 0, 0])
            std = torch.tensor([1, 1, 1])
        else:
            mean = torch.tensor([0.4534, 0.3770, 0.3885])
            std = torch.tensor([0.2512, 0.1881, 0.2599])

        self.image_preprocessor = T.Compose([
            T.ToTensor(),
            T.Resize((image_size, image_size)),
            T.Normalize(
                mean = mean,
                std = std  # To be computed
            )
        ])

        # Loading all the images in the images vector
        print("Loading all the images: \n")
        for demo_images_path in tqdm(image_path_list):
            demo_image_folder_path = os.path.join(images_dir, demo_images_path)

            # Sort the demo images list
            demo_images_list = os.listdir(demo_image_folder_path)
            demo_images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            # Read each image and append them in the images array
            for idx in range(len(demo_images_list) - 1):
                try:
                    image = Image.open(os.path.join(demo_image_folder_path, demo_images_list[idx]))
                    image = image.crop((220, 165, 460, 340)) # Left, Top, Right, Bottom
                    image_tensor = self.image_preprocessor(image)
                    self.image_tensors.append(image_tensor.detach())
                    image.close()
                except:
                    print('Image cannot be read!')
                    continue

        # Loading all the action vectors
        print("\nLoading all the actions: \n")
        for demo_action_path in tqdm(action_path_list):
            demo_actions = torch.load(os.path.join(actions_dir, demo_action_path))
            self.action_tensors.append(demo_actions)
        
        self.action_tensors = torch.cat(self.action_tensors, dim = 0)
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return ((self.image_tensors[idx], self.action_tensors[idx]))

class FidgetSpinningVisualDataset(Dataset):
    def __init__(self, type = None, image_size = 224, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/fidget_spinning'):
        if type == "train":
            self.data_path = os.path.join(data_path, "for_eval", "train")
        elif type == "val":
            self.data_path = os.path.join(data_path, "for_eval", "validation")
        elif type == "test":
            self.data_path = os.path.join(data_path, "for_eval", "test")
        else:
            self.data_path = os.path.join(data_path, "complete")

        # Loading all the image and action folder paths
        images_dir, actions_dir = os.path.join(self.data_path, 'images'), os.path.join(self.data_path, 'actions')
        
        # Sorting all the demo paths
        image_path_list, action_path_list = os.listdir(images_dir), os.listdir(actions_dir)

        image_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        action_path_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.image_tensors, self.action_tensors = [], []

        # Transforming the images
        if type == "train":
            mean = torch.tensor([0.4631, 0.4917, 0.5201]) # To be recalculated
            std = torch.tensor([0.2885, 0.2666, 0.2526])  # To be recalculated
        elif type == "val":
            mean = torch.tensor([0.3557, 0.4138, 0.4550]) # To be recalculated
            std = torch.tensor([0.9030, 0.8565, 0.8089])  # To be recalculated
        elif type == "load":
            mean = torch.tensor([0, 0, 0])
            std = torch.tensor([1, 1, 1])
        else:
            mean = torch.tensor([0.4320, 0.3963, 0.3478])
            std = torch.tensor([0.2897, 0.2525, 0.2317])

        self.image_preprocessor = T.Compose([
            T.ToTensor(),
            T.Resize((image_size, image_size)),
            T.Normalize(
                mean = mean, 
                std = std  
            )
        ])

        # Loading all the images in the images vector
        print("Loading all the images: \n")
        for demo_images_path in tqdm(image_path_list):
            demo_image_folder_path = os.path.join(images_dir, demo_images_path)

            # Sort the demo images list
            demo_images_list = os.listdir(demo_image_folder_path)
            demo_images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            # print("Reading images from {}".format(demo_image_folder_path))

            # Read each image and append them in the images array
            for idx in range(len(demo_images_list) - 1):
                try:
                    image = Image.open(os.path.join(demo_image_folder_path, demo_images_list[idx]))
                    image = image.crop((65, 80, 590, 480)) # Left, Top, Right, Bottom
                    image_tensor = self.image_preprocessor(image)
                    self.image_tensors.append(image_tensor.detach())
                    image.close()
                except:
                    print('Image cannot be read!')
                    continue

        # Loading all the action vectors
        print("\nLoading all the actions: \n")
        for demo_action_path in tqdm(action_path_list):
            demo_actions = torch.load(os.path.join(actions_dir, demo_action_path))
            self.action_tensors.append(demo_actions)
        
        self.action_tensors = torch.cat(self.action_tensors, dim = 0)
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return ((self.image_tensors[idx], self.action_tensors[idx]))

if __name__ == '__main__':

    # To find the mean and std of the image pixels for normalization
    # dataset = CubeRotationVisualDataset()
    # dataset = ObjectFlippingVisualDataset()
    # dataset = FidgetSpinningVisualDataset()
    dataset = RepresentationVisualDataset("fidget_spinning")
    print("Number of images in the dataset: {}\n".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size = 128, shuffle = False, pin_memory = True, num_workers = 24)

    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for images, actions in tqdm(dataloader):
        psum += images.sum(axis = [0, 2, 3])
        psum_sq += (images ** 2).sum(axis = [0, 2, 3])

    count = len(dataset) * 224 * 224

    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    print("Mean: {}".format(total_mean))
    print("Std: {}".format(total_std))