"""
    Setsup the data for the model
"""
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from torchvision.models import DenseNet121_Weights
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (str): Path to the directory containing class folders.
            transform (callable): Transformations to apply to the images.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Folder names as class labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Map folder name to class index
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_path = self.root_dir / Path(cls)
            for img_file in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img_file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(root_dir, batch_size=32):
    """
    Initialize the data loaders for the ChestXray dataset

    Args:
        root_dir (str): Path to the directory containing the train, test, and valid folders.
        batch_size (int): Batch size for the data loaders.

    Returns:
        train_dataloader (DataLoader): Data loader for the training set.
        test_dataloader (DataLoader): Data loader for the test set.
        valid_dataloader (DataLoader): Data loader for the validation set.
        class_names (list): List of class names.

    """
    # Specify the weights to use from the DenseNet121 model
    weights = DenseNet121_Weights.DEFAULT
    # Get the transforms associated with the weights
    densenet_transforms = weights.transforms()

    # Initialize datasets
    train_dataset = ChestXrayDataset(root_dir=os.path.join(root_dir, "train"), transform=densenet_transforms)
    test_dataset = ChestXrayDataset(root_dir=os.path.join(root_dir, "test"), transform=densenet_transforms)
    valid_dataset = ChestXrayDataset(root_dir=os.path.join(root_dir, "valid"), transform=densenet_transforms)

    # Initialize data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, valid_dataloader, train_dataset.classes
