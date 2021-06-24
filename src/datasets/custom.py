from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Subset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.transforms import Normalize
from .preprocessing import create_semisupervised_setting
from base.base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import numpy as np

import torch
import torchvision.transforms as transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
class Custom_Dataset(BaseADDataset):

    def __init__(self, root: str):
        super().__init__(root)

        # Define normal and outlier classes
       # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)
        self.known_outlier_classes = (1,)
        
        #TODO: check if 0.5 at everything works better
        transform = transforms.Compose([transforms.ToTensor(), 
                                        #transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225],
        #),
                                       ])
        target_transform = transforms.Lambda(lambda x: -1 if x == 1 else 1 if x==0 else None)

        # Get train set
        self.train_set = CustomDataset(root=self.root+"/train", transform=transform, target_transform=target_transform)
        # Create semi-supervised setting
        #idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
        #                                                     self.outlier_classes, self.known_outlier_classes,
        #                                                     ratio_known_normal, ratio_known_outlier, ratio_pollution)
        #train_set.semi_targets = torch.tensor(train_set.targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        #self.train_set = Subset(train_set, idx)
        

        # Get test set
        self.test_set = CustomDataset(root=self.root+"/test", transform=transform, target_transform=target_transform)
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                    num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                    num_workers=num_workers, drop_last=False)
        return train_loader, test_loader



class CustomDataset(DatasetFolder):
    """
    Custom Dataset for Deep-SAD

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """
    
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None,):
        super(CustomDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.semi_targets = torch.tensor(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        semi_t = self.semi_targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            print("before:",str(semi_t))
            semi_t = self.target_transform(semi_t)
            print("after:",str(semi_t))

        return img, target, self.semi_targets[index], index


    def __len__(self) -> int:
        return len(self.imgs)
