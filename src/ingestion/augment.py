import os
import cv2
from pathlib import Path

def augment_and_label():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    categories = {'ok_front': 0, 'def_front': 1}
    
    for cat_name, label in categories.items():
        source_path = raw_dir / cat_name
        # Create subfolders in processed for cleanliness
        target_path = processed_dir / cat_name
        target_path.mkdir(exist_ok=True)

        for img_name in os.listdir(source_path):
            img = cv2.imread(str(source_path / img_name))
            if img is None: continue # Validation: skip corrupt

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(target_path / f"orig_{img_name}"), gray_img)

            # Augmentation: Horizontal Flip
            flipped = cv2.flip(gray_img, 1)
            cv2.imwrite(str(target_path / f"flip_{img_name}"), flipped)

            # Augmentation: Rotation (90 deg)
            rotated = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(target_path / f"rot_{img_name}"), rotated)

    print("Augmentation and Labeling complete.")

if __name__ == "__main__":
    augment_and_label()