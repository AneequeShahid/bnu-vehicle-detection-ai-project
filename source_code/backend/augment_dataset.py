import cv2
import numpy as np
import os
import glob

# ============ CONFIG ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "dataset", "extracted", "train")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

def rotate_image_and_boxes(image, boxes, angle_deg):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    new_boxes = []
    for box in boxes:
        cls_id, x_c, y_c, box_w, box_h = box
        x_c_px, y_c_px = x_c * w, y_c * h
        box_w_px, box_h_px = box_w * w, box_h * h
        
        x1, y1 = x_c_px - box_w_px / 2.0, y_c_px - box_h_px / 2.0
        x2, y2 = x_c_px + box_w_px / 2.0, y_c_px - box_h_px / 2.0
        x3, y3 = x_c_px + box_w_px / 2.0, y_c_px + box_h_px / 2.0
        x4, y4 = x_c_px - box_w_px / 2.0, y_c_px + box_h_px / 2.0
        
        corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        ones = np.ones(shape=(len(corners), 1))
        points_ones = np.hstack((corners, ones))
        transformed_corners = M.dot(points_ones.T).T
        
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        
        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)
        
        xmin = max(0, min(xmin, new_w))
        xmax = max(0, min(xmax, new_w))
        ymin = max(0, min(ymin, new_h))
        ymax = max(0, min(ymax, new_h))
        
        new_x_c = (xmin + xmax) / 2.0 / new_w
        new_y_c = (ymin + ymax) / 2.0 / new_h
        new_box_w = (xmax - xmin) / new_w
        new_box_h = (ymax - ymin) / new_h
        
        new_boxes.append((cls_id, new_x_c, new_y_c, new_box_w, new_box_h))
        
    return rotated_image, new_boxes

def flip_image_and_boxes(image, boxes):
    flipped_image = cv2.flip(image, 1)
    new_boxes = []
    for box in boxes:
        cls_id, x_c, y_c, box_w, box_h = box
        # Flip x coordinate: new_x_center = 1.0 - old_x_center
        new_boxes.append((cls_id, 1.0 - x_c, y_c, box_w, box_h))
    return flipped_image, new_boxes

def add_gaussian_noise(image, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


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
    print("[INFO] Starting Dataset Augmentation to 1000+ images...")
    
    # We want to search for original images first (skip already augmented ones)
    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    original_images = [p for p in image_paths if not any(x in os.path.basename(p) for x in ["_rot_", "_flip_", "_bright_", "_dark_", "_noise_", "_blur_"])]
    
    print(f"[INFO] Found {len(original_images)} original images.")
    
    angles = [5, -5, 10, -10, 15, -15]
    total_generated = 0
    
    for img_path in original_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        boxes = load_yolo_labels(label_path)
        
        # 1. Apply rotations (6 versions)
        for angle in angles:
            suffix = f"_rot_{angle}" if angle > 0 else f"_rot_neg{abs(angle)}"
            aug_img_path = os.path.join(IMAGES_DIR, f"{base_name}{suffix}.jpg")
            aug_lbl_path = os.path.join(LABELS_DIR, f"{base_name}{suffix}.txt")
            
            if not os.path.exists(aug_img_path):
                rot_img, rot_boxes = rotate_image_and_boxes(image, boxes, angle)
                cv2.imwrite(aug_img_path, rot_img)
                save_yolo_labels(aug_lbl_path, rot_boxes)
                total_generated += 1
                
        # 2. Apply horizontal flip (1 version)
        flip_img_path = os.path.join(IMAGES_DIR, f"{base_name}_flip.jpg")
        flip_lbl_path = os.path.join(LABELS_DIR, f"{base_name}_flip.txt")
        if not os.path.exists(flip_img_path):
            flip_img, flip_boxes = flip_image_and_boxes(image, boxes)
            cv2.imwrite(flip_img_path, flip_img)
            save_yolo_labels(flip_lbl_path, flip_boxes)
            total_generated += 1
            
        # 3. Apply brightness adjustments (4 versions: +20, -20, +40, -40)
        for val in [20, -20, 40, -40]:
            suffix = f"_bright_{val}" if val > 0 else f"_dark_{abs(val)}"
            bright_img_path = os.path.join(IMAGES_DIR, f"{base_name}{suffix}.jpg")
            bright_lbl_path = os.path.join(LABELS_DIR, f"{base_name}{suffix}.txt")
            
            if not os.path.exists(bright_img_path):
                # Adjust brightness
                aug_img = cv2.convertScaleAbs(image, alpha=1.0, beta=val)
                cv2.imwrite(bright_img_path, aug_img)
                save_yolo_labels(bright_lbl_path, boxes) # boxes remain same
                total_generated += 1

        # 4. Apply Gaussian Noise (2 versions: sigma=15, sigma=25)
        for sigma in [15, 25]:
            noise_img_path = os.path.join(IMAGES_DIR, f"{base_name}_noise_{sigma}.jpg")
            noise_lbl_path = os.path.join(LABELS_DIR, f"{base_name}_noise_{sigma}.txt")
            if not os.path.exists(noise_img_path):
                noise_img = add_gaussian_noise(image, sigma=sigma)
                cv2.imwrite(noise_img_path, noise_img)
                save_yolo_labels(noise_lbl_path, boxes)
                total_generated += 1

        # 5. Apply Gaussian Blur (2 versions: 3x3, 5x5)
        for ksize in [(3, 3), (5, 5)]:
            suffix = f"_blur_{ksize[0]}x{ksize[1]}"
            blur_img_path = os.path.join(IMAGES_DIR, f"{base_name}{suffix}.jpg")
            blur_lbl_path = os.path.join(LABELS_DIR, f"{base_name}{suffix}.txt")
            if not os.path.exists(blur_img_path):
                blur_img = apply_gaussian_blur(image, kernel_size=ksize)
                cv2.imwrite(blur_img_path, blur_img)
                save_yolo_labels(blur_lbl_path, boxes)
                total_generated += 1
                
    # Final count check
    all_final_images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    print(f"[SUCCESS] Generated {total_generated} new augmented images.")
    print(f"[INFO] Total images in dataset: {len(all_final_images)}")

if __name__ == '__main__':
    main()
