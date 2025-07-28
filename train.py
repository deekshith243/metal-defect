import torch
from torch.utils.data import DataLoader
from custom_dataset import MetalDefectDataset, get_transform, collate_fn
from model import get_faster_rcnn_model
import math
import sys
import os
import utils # Import your utils.py

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

if __name__ == '__main__':
    # Define paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset") # Make sure this matches your dataset root

    IMAGE_TRAIN_DIR = os.path.join(DATA_DIR, "images", "train")
    ANNOTATION_TRAIN_PATH = os.path.join(DATA_DIR, "annotations", "instances_train.json")
    IMAGE_VAL_DIR = os.path.join(DATA_DIR, "images", "val")
    ANNOTATION_VAL_PATH = os.path.join(DATA_DIR, "annotations", "instances_val.json")

    # Define number of classes (count your unique defect classes + 1 for background)
    # E.g., if you have "scratches", "dents", "pits", "cuts" (4 defect classes)
    NUM_CLASSES = 4 + 1 # Adjust this based on YOUR specific defect classes from Roboflow

    # Hyperparameters - You might need to tune these later
    BATCH_SIZE = 2 # Adjust based on your GPU memory
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    PRINT_FREQ = 10 # Print loss every N iterations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the dataset
    # Ensure your dataset/images/train and dataset/annotations/instances_train.json exist
    try:
        train_dataset = MetalDefectDataset(IMAGE_TRAIN_DIR, ANNOTATION_TRAIN_PATH, get_transform(train=True))
        val_dataset = MetalDefectDataset(IMAGE_VAL_DIR, ANNOTATION_VAL_PATH, get_transform(train=False))
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your 'dataset' folder is organized correctly with 'images/train', 'images/val', and 'annotations/instances_train.json', 'annotations/instances_val.json'.")
        sys.exit(1)


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(), # Use all available CPU cores for data loading
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Validation batch size can be smaller
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn
    )

    # 2. Get the model
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.to(device)

    # 3. Define Optimizer and Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler (optional but recommended)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Train
        train_one_epoch(model, optimizer, train_loader, device, epoch, PRINT_FREQ)
        lr_scheduler.step()

        # Save the model
        model_save_path = f"faster_rcnn_metal_defect_epoch_{epoch:02d}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    print("Training complete!")