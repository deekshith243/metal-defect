import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as T

class MetalDefectDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)

        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.annotations_by_image_id = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[img_id] = []
            self.annotations_by_image_id[img_id].append(ann)

        self.category_id_to_label = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.label_to_category_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        annotations = self.annotations_by_image_id.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            image = self.transforms(image)

        return image, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Add data augmentation for training
        transforms.append(T.RandomHorizontalFlip(0.5))
        # You can add more augmentations here (e.g., RandomAffine, ColorJitter)
    return T.Compose(transforms)

# Helper function for DataLoader collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))