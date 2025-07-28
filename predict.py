import torch
from PIL import Image
import torchvision.transforms as T
from model import get_faster_rcnn_model # Assuming your model function is in model.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def predict(model_path, image_path, num_classes, threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # Add Normalize if you added it during training
    image_tensor = transform(image).to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        prediction = model([image_tensor])

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Filter out predictions below a certain confidence threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return boxes, labels, scores, image

def visualize_predictions(image, boxes, labels, scores, class_names=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        label = labels[i]
        score = scores[i]

        # Adjust label index if your class_names list is 0-indexed and model labels start from 1 (background is 0)
        # Assuming your labels from Roboflow are 1-indexed (e.g., pits=1, scratches=2, etc.) and background is 0
        display_label = ""
        if class_names and label > 0 and label <= len(class_names):
            display_label = f"{class_names[label-1]}: {score:.2f}"
        else:
            display_label = f"Class {label}: {score:.2f}"

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x_min, y_min - 5, display_label, color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # REPLACE THESE WITH YOUR ACTUAL VALUES
    MODEL_PATH = "faster_rcnn_metal_defect_epoch_09.pth" # Path to your saved model (e.g., the last one)
    TEST_IMAGE_PATH = "dataset/images/val/a_new_test_image.jpg" # Path to an image NOT in your training/validation set

    # Define the actual number of defect classes (excluding background)
    # This should match the NUM_CLASSES - 1 you used in train.py
    NUM_DEFECT_CLASSES = 4 # e.g., if you had 'scratches', 'dents', 'pits', 'cuts'

    # A list of your defect class names in the order of their Roboflow category IDs
    # If Roboflow assigned 'pits' as ID 1, 'scratches' as ID 2, etc. (after background 0)
    # You'll need to confirm the order from your Roboflow project's categories.json or your memory.
    CLASS_NAMES = ['pits', 'scratches', 'dents', 'cuts'] # Example order, verify from your Roboflow project

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please ensure training completed and the path is correct.")
    elif not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}.")
    else:
        # NUM_CLASSES for predict function is actually NUM_DEFECT_CLASSES + 1
        boxes, labels, scores, original_image = predict(MODEL_PATH, TEST_IMAGE_PATH, NUM_DEFECT_CLASSES + 1)
        print("Detected objects:")
        for i in range(len(boxes)):
            class_name = CLASS_NAMES[labels[i]-1] if labels[i] > 0 else "Background" # Adjust for 0-indexed class_names
            print(f"  Box: {boxes[i]}, Label: {class_name}, Score: {scores[i]:.2f}")
        visualize_predictions(original_image, boxes, labels, scores, CLASS_NAMES)