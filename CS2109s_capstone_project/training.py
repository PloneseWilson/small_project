from psutil import net_connections
import os
import numpy as np
import json
import random

from IPython.display import display

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


ASSET_PATH = "/content/data/data/assets"

BG_LABELS = ["floor", "lava", "wall"]
CT_LABELS = ["floor", "human", "exit", "box", "locked", "gem", "key", "coin", "boots", "shield", "ghost"]
# "opened" not here

def average_pool_2d(image_array, size=3):
    h, w, c = image_array.shape
    new_h, new_w = h // size, w // size

    reshaped = image_array[:new_h*size, :new_w*size, :].reshape(new_h, size, new_w, size, c)

    # 3. Mean across the window (axis 1 and 3)
    return reshaped.mean(axis=(1, 3))

def extract_features(tile_pixels):
    img = Image.fromarray(tile_pixels[:, :, :3].astype(np.uint8))
    tile_rgb = np.array(img.resize((48, 48)))

    bg_feat = np.concatenate([tile_rgb[:1, :, :], tile_rgb[-1:, :, :]], axis=0).flatten()

    shredded_ct = tile_rgb[5:-5, 5:-5, :]
    pooled_ct = average_pool_2d(shredded_ct, size=3)
    ct_feat = pooled_ct.flatten()

    return bg_feat, ct_feat

def get_overlapped_features(img, floor):
    background = random.choice(floor).copy().convert('RGB')
    tile_w, tile_h = background.size

    rgba = img.convert('RGBA')
    obj_w, obj_h = rgba.size


    offset_x = (tile_w - obj_w) // 2
    offset_y = (tile_h - obj_h) // 2

    background.paste(rgba, (offset_x, offset_y), rgba)
    overlap_array = np.array(background.convert('RGB'))

    display(background)

    return extract_features(overlap_array)

def load_and_split_data(root):
    t_bg_X, t_bg_y, v_bg_X, v_bg_y = [], [], [], []
    t_ct_X, t_ct_y, v_ct_X, v_ct_y = [], [], [], []

    # Get only valid directories that are in our label lists
    valid_folders = [d for d in os.listdir(root) if d in BG_LABELS or d in CT_LABELS]
    print(f"Found {len(valid_folders)} valid asset folders.")

    floor_dir = os.path.join(root, "floor")
    floor_files = [f for f in os.listdir(floor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    floor_images = [Image.open(os.path.join(floor_dir, f)).convert('RGB') for f in floor_files]

    for folder in valid_folders:
        folder_path = os.path.join(root, folder)

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not files:
            continue

        print(f"Processing {folder}... ({len(files)} images)")

        for idx, img_name in enumerate(files):
          img = Image.open(os.path.join(folder_path, img_name)).convert('RGBA')
          pixels = np.array(img)

          ct_feat = None
          bg_feat, _ = extract_features(pixels)

          if folder in CT_LABELS and (folder != "floor" and folder != "lava" and folder != "wall"):
              _, ct_feat = get_overlapped_features(img, floor_images)

          is_train = (idx % 5 != 4)

          if folder in BG_LABELS:
            label = BG_LABELS.index(folder)
            if is_train: t_bg_X.append(bg_feat); t_bg_y.append(label)
            else: v_bg_X.append(bg_feat); v_bg_y.append(label)

          if folder in CT_LABELS:
                label = CT_LABELS.index(folder)

                # If it's a floor/lava/wall, just use raw
                if folder in BG_LABELS:
                    _, ct_feat = extract_features(pixels)
                    if is_train: t_ct_X.append(ct_feat); t_ct_y.append(label)
                    else: v_ct_X.append(ct_feat); v_ct_y.append(label)
                else:
                    num_variants = 5 if is_train else 1
                    for _ in range(num_variants):
                        _, ct_feat = get_overlapped_features(img, floor_images)

                        if is_train:
                            t_ct_X.append(ct_feat)
                            t_ct_y.append(label)
                        else:
                            v_ct_X.append(ct_feat)
                            v_ct_y.append(label)


    return (np.array(t_bg_X), np.array(t_bg_y), np.array(v_bg_X), np.array(v_bg_y),
            np.array(t_ct_X), np.array(t_ct_y), np.array(v_ct_X), np.array(v_ct_y))

# --- Run the process ---
results = load_and_split_data(ASSET_PATH)
tr_bg_X, tr_bg_y, ts_bg_X, ts_bg_y, tr_ct_X, tr_ct_y, ts_ct_X, ts_ct_y = results

if tr_bg_X is not None and len(tr_bg_X) > 0:
    print("\nTraining Models...")

    # Background Model
    clf_bg = LogisticRegression(max_iter=3000).fit(tr_bg_X, tr_bg_y)
    print(f"BG Accuracy: {accuracy_score(ts_bg_y, clf_bg.predict(ts_bg_X)):.4f}")

    # Center Model
    clf_ct = LogisticRegression(max_iter=3000).fit(tr_ct_X, tr_ct_y)
    print(f"Center Accuracy: {accuracy_score(ts_ct_y, clf_ct.predict(ts_ct_X)):.4f}")

    # Create a dictionary to hold all your math
    model_data = {
      "bg_weights": np.round(clf_bg.coef_,6).tolist(),
      "bg_intercept": np.round(clf_bg.intercept_,6).tolist(),
      "ct_weights": np.round(clf_ct.coef_,6).tolist(),
      "ct_intercept": np.round(clf_ct.intercept_,6).tolist(),
    }
    with open("final_weights.txt", "w") as f:
      json.dump(model_data, f)
else:
    print("\nFAILED: No data loaded. Check if ASSET_PATH is exactly correct.")
