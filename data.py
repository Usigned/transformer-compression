import torch
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor, Normalize

VIT_CIFAR10_NORM = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

VIT_CIFAR10_TFM = transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    VIT_CIFAR10_NORM,
])

def get_cifar10_dataloader(root='./data', train=True, transforms=VIT_CIFAR10_TFM, batch_size=32, num_workers=2, **kwargs):
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=False,
        transform=transforms
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train, **kwargs)