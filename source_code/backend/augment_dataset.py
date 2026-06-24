import cv2
import numpy as np
import os
import glob
import math

# ============ CONFIG ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "dataset", "extracted", "train")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

def rotate_image_and_boxes(image, boxes, angle_deg):
    """
    Rotates an image and its corresponding YOLO bounding boxes by angle_deg.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    
    # Calculate new image dimensions to avoid clipping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix translation
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    
    # Rotate the image
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    new_boxes = []
    for box in boxes:
        cls_id, x_c, y_c, box_w, box_h = box
        
        # Convert normalized coordinates to absolute pixels
        x_c_px, y_c_px = x_c * w, y_c * h
        box_w_px, box_h_px = box_w * w, box_h * h
        
        # Define the 4 corners of the bounding box
        x1, y1 = x_c_px - box_w_px / 2.0, y_c_px - box_h_px / 2.0
        x2, y2 = x_c_px + box_w_px / 2.0, y_c_px - box_h_px / 2.0
        x3, y3 = x_c_px + box_w_px / 2.0, y_c_px + box_h_px / 2.0
        x4, y4 = x_c_px - box_w_px / 2.0, y_c_px + box_h_px / 2.0
        
        corners = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ])
        
        # Transform the corners using the rotation matrix
        ones = np.ones(shape=(len(corners), 1))
        points_ones = np.hstack((corners, ones))
        transformed_corners = M.dot(points_ones.T).T
        
        # Find the new bounding box (min/max of rotated corners)
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        
        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)
        
        # Clip to new image boundaries
        xmin = max(0, min(xmin, new_w))
        xmax = max(0, min(xmax, new_w))
        ymin = max(0, min(ymin, new_h))
        ymax = max(0, min(ymax, new_h))
        
        # Convert back to YOLO normalized format
        new_x_c = (xmin + xmax) / 2.0 / new_w
        new_y_c = (ymin + ymax) / 2.0 / new_h
        new_box_w = (xmax - xmin) / new_w
        new_box_h = (ymax - ymin) / new_h
        
        new_boxes.append((cls_id, new_x_c, new_y_c, new_box_w, new_box_h))
        
    return rotated_image, new_boxes

def load_yolo_labels(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    boxes.append((cls_id, *coords))
    return boxes

def save_yolo_labels(label_path, boxes):
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def main():
    print("[INFO] Starting Dataset Augmentation...")
    
    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    print(f"[INFO] Found {len(image_paths)} original images.")
    
    angles = [5, -5, 10, -10]
    total_generated = 0
    
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        
        # Load image and labels
        image = cv2.imread(img_path)
        if image is None:
            continue
        boxes = load_yolo_labels(label_path)
        
        # Apply rotations
        for angle in angles:
            suffix = f"_rot_{angle}" if angle > 0 else f"_rot_neg{abs(angle)}"
            aug_img_name = f"{base_name}{suffix}.jpg"
            aug_lbl_name = f"{base_name}{suffix}.txt"
            
            aug_img_path = os.path.join(IMAGES_DIR, aug_img_name)
            aug_lbl_path = os.path.join(LABELS_DIR, aug_lbl_name)
            
            # Skip if already exists
            if os.path.exists(aug_img_path):
                continue
                
            rot_img, rot_boxes = rotate_image_and_boxes(image, boxes, angle)
            
            # Save augmented image and label
            cv2.imwrite(aug_path := aug_img_path, rot_img)
            save_yolo_labels(aug_lbl_path, rot_boxes)
            total_generated += 1
            
    print(f"[SUCCESS] Augmented {total_generated} new images. Total dataset size increased!")

if __name__ == '__main__':
    main()
