import torch
from torchvision import transforms as T

def augmentation_generator(task):
    if task == "rotate":
        norm_values = T.Normalize(
            mean = torch.tensor([0.3484, 0.3638, 0.3819]), 
            std = torch.tensor([0.3224, 0.3151, 0.3166])  
        )
    elif task == "flip":
        norm_values = T.Normalize(
            mean = torch.tensor([0, 0, 0]), 
            std = torch.tensor([1, 1, 1])  
        )
    elif task == "spin":
        norm_values = T.Normalize(
            mean = torch.tensor([0, 0, 0]), 
            std = torch.tensor([1, 1, 1])  
        )

    augment_custom =  T.Compose([
        T.RandomResizedCrop(224, scale = (0.6, 1)),  
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
        norm_values
    ])

    return augment_custom