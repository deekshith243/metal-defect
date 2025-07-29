import torch
from PIL import Image
import torchvision.transforms as T
from model import get_faster_rcnn_model # Assuming your model function is in model.py
import cv2 # Import OpenCV for camera capture and display
import os
import time # For potential frame rate logging

# (Optional) You might want to define color for each class if visualizing
COLORS = {
    'pits': (0, 255, 255),    # Yellow-ish (BGR)
    'scratches': (0, 0, 255), # Red
    'dents': (255, 0, 0),     # Blue
    'cuts': (0, 255, 0)      # Green
}

def load_model(model_path, num_classes):
    """Loads the Faster R-CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Will default to 'cpu' on Pi
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    model.to(device)
    return model, device

def preprocess_frame(frame, transform, device):
    """Preprocesses a single OpenCV frame for model inference."""
    # Convert OpenCV BGR image to PIL RGB image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = transform(image_pil).to(device)
    return image_tensor

def perform_inference(model, image_tensor, threshold=0.7):
    """Performs inference on the preprocessed tensor."""
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

    return boxes, labels, scores

def visualize_predictions_on_frame(frame, boxes, labels, scores, class_names=None, colors=None):
    """
    Visualizes predictions on an OpenCV image frame.
    Args:
        frame: The OpenCV image frame (NumPy array, BGR).
        boxes: Detected bounding boxes.
        labels: Predicted labels.
        scores: Prediction scores.
        class_names: List of class names.
        colors: Dictionary mapping class names to BGR tuples.
    Returns:
        frame_with_detections: The frame with bounding boxes and labels drawn.
    """
    frame_with_detections = frame.copy()

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        label_id = labels[i]
        score = scores[i]

        display_label = ""
        color = (0, 0, 255) # Default Red (BGR)
        
        if class_names and label_id > 0 and label_id <= len(class_names):
            class_name = class_names[label_id-1]
            display_label = f"{class_name}: {score:.2f}"
            if colors and class_name in colors:
                color = colors[class_name]
        else:
            display_label = f"Class {label_id}: {score:.2f}"

        # Draw bounding box
        cv2.rectangle(frame_with_detections, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label and score text
        # Get text size to draw a background rectangle for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame_with_detections, (x_min, y_min - text_height - 5), (x_min + text_width, y_min), color, -1) # Background for text
        cv2.putText(frame_with_detections, display_label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text

    return frame_with_detections

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Adjust MODEL_PATH to where your .pth model is located on the Raspberry Pi
    # If you copied it to the same directory as this script:
    MODEL_PATH = "faster_rcnn_metal_defect_epoch_09.pth"
    # If you put it in a 'models' subfolder:
    # MODEL_PATH = "models/faster_rcnn_metal_defect_epoch_09.pth"

    NUM_DEFECT_CLASSES = 4 # e.g., 'pits', 'scratches', 'dents', 'cuts'
    CLASS_NAMES = ['pits', 'scratches', 'dents', 'cuts']
    CONFIDENCE_THRESHOLD = 0.7 # Adjust as needed

    # Camera settings for Raspberry Pi Camera Module 3
    # CAMERA_ID = 0 is usually the correct index for the official Pi Camera Module
    CAMERA_ID = 0 
    
    # Optional: Set camera resolution. 12MP implies higher native resolutions,
    # but downscaling for inference can improve speed.
    # Set this to a resolution your model was trained on or can handle efficiently.
    # Common choices might be 640x480, 1280x720, or even 1920x1080 if performance allows.
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # --- Setup Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        print("Please ensure the model file is transferred to your Raspberry Pi")
        print("and the MODEL_PATH variable in this script is correct.")
        exit()

    model, device = load_model(MODEL_PATH, NUM_DEFECT_CLASSES + 1)
    print(f"Model loaded successfully on {device}.")

    # Define the same transformations as used during training
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Setup Camera Capture ---
    # The default backend usually works for Pi Camera Module via libcamera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Set camera resolution. This is important to control input size.
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        print(f"Attempting to set camera resolution to {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        # Verify if the resolution was actually set (cameras might not support all requests)
        print(f"Actual camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")


    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {CAMERA_ID}.")
        print("Please check camera connection and 'raspi-config' settings for camera enablement.")
        print("Also ensure necessary camera drivers/backends are installed (e.g., libcamera).")
        exit()
    else:
        print(f"Camera with ID {CAMERA_ID} opened successfully.")


    print("Starting live detection. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read() # Read a frame from the camera
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # --- Perform Detection ---
            # Measure inference time (optional)
            # inference_start_time = time.time()
            image_tensor = preprocess_frame(frame, transform, device)
            boxes, labels, scores = perform_inference(model, image_tensor, CONFIDENCE_THRESHOLD)
            # inference_end_time = time.time()
            # print(f"Inference time: {inference_end_time - inference_start_time:.4f} seconds")


            # --- Visualize Results ---
            frame_with_detections = visualize_predictions_on_frame(frame, boxes, labels, scores, CLASS_NAMES, COLORS)

            # --- Display Frame ---
            cv2.imshow('Metal Defect Detection (Live)', frame_with_detections)

            # --- Optional: Print Detection Info to Console ---
            # You might want to limit this or log to a file for production
            if len(boxes) > 0:
                # print(f"\nDetected objects in frame {frame_count}:")
                for i in range(len(boxes)):
                    class_name = CLASS_NAMES[labels[i]-1] if labels[i] > 0 else "Background"
                    # print(f"  Box: {boxes[i].astype(int)}, Label: {class_name}, Score: {scores[i]:.2f}")
            
            frame_count += 1
            # Calculate and print FPS every N frames (for performance monitoring)
            if frame_count % 30 == 0: # Every 30 frames
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = time.time() # Reset timer for the next 30 frames

            # --- Exit Condition ---
            # Press 'q' to quit the live stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" 'q' pressed. Exiting.")
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # --- Clean Up ---
        cap.release() # Release the camera resource
        cv2.destroyAllWindows() # Close all OpenCV display windows
        print("Camera released and windows closed.")