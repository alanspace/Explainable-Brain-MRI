import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir=None, image_paths=None, labels=None, transform=None, phase='train'):
        """
        Args:
            root_dir (string, optional): Directory with all the images.
            image_paths (list, optional): List of absolute paths to images.
            labels (list, optional): List of integer labels corresponding to image_paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            phase (string): 'train' or 'test/val'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        # Standard Brain Tumor Classes (Alphabetical order usually)
        self.classes = sorted(['glioma', 'meningioma', 'notumor', 'pituitary'])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        elif root_dir is not None:
            self.image_paths = []
            self.labels = []
            self._load_dataset()
        else:
            raise ValueError("Either root_dir OR (image_paths and labels) must be provided.")

    def _load_dataset(self):
        # Assumes structure: root_dir/class_name/*.jpg (or png, jpeg)
        # Using glob to find files recursively if needed
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(class_dir):
                # Try case insensitive match or flexible naming if needed
                # For now, warn and skip
                print(f"Warning: Directory {class_dir} not found. Skipping class {cls_name}.")
                continue
                
            # Grab images
            valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
            for ext in valid_exts:
                files = glob(os.path.join(class_dir, ext))
                for f in files:
                    self.image_paths.append(f)
                    self.labels.append(self.class_to_idx[cls_name])
        
        print(f"[{self.phase.upper()}] Loaded {len(self.image_paths)} images from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)
