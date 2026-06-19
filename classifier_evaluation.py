import numpy as np
import os
import json
import re
import cv2
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from coral_obj_classifier_net import Boost_Classifier

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coral masks for spectral dataset using classifier and SAM.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root directory")
    args = parser.parse_args()

    PUB_DATA_ROOT=args.data_path

    CLASSIFIER_PATH = PUB_DATA_ROOT + "network_models/classifier.pth"
    OUTPUT_DIR = PUB_DATA_ROOT + "eval_classifier/"

    SAVE_VIS = True  # set False to skip figure generation

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 16
    num_classes = 3
    spectrum_min = 400.0
    spectrum_max = 700.0

    CLASS_NAMES  = ["healthy", "sick", "others"]
    CLASS_COLORS = ["#2ca02c", "#d62728", "#1f77b4"]  # green, red, blue

    #---- wavelengths
    dataset_wavelengths = np.fromfile(PUB_DATA_ROOT + "entry_0000/raw_data/wavelengths.bin", dtype=np.float32)
    wavelength_indices = np.where((dataset_wavelengths >= spectrum_min) & (dataset_wavelengths <= spectrum_max))[0]
    main_spectra = dataset_wavelengths[wavelength_indices]

    #---- classifier
    classifier = Boost_Classifier(latent_dim, num_classes, main_spectra)
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=torch_device, weights_only=False)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    classifier.to(torch_device)
    for param in classifier.parameters():
        param.requires_grad = False
    print("Classifier loaded.")


    def description_to_class(description: str) -> int:
        desc = description.lower()
        if "sick" in desc or "unhealthy" in desc or "coral_holder" in desc:
            return 1
        if "healthy" in desc:
            return 0
        return 2


    def parse_gt_pixels(meta_info):
        """Return (gt_yx, gt_classes) arrays from meta_data.json labels, or (None, None) if no pixels."""
        yx_list, class_list = [], []
        for label in meta_info.get("labels", []):
            pixels_raw = label.get("pixels", "[]")
            pixels = json.loads(pixels_raw) if isinstance(pixels_raw, str) else pixels_raw
            if not pixels:
                continue
            cls = description_to_class(label.get("description", ""))
            for yx in pixels:
                yx_list.append(yx)
                class_list.append(cls)
        if not yx_list:
            return None, None
        yx_arr = np.array(yx_list, dtype=np.int32)
        cls_arr = np.array(class_list, dtype=np.int32)
        unique_yx, unique_idx, counts = np.unique(yx_arr, axis=0, return_index=True, return_counts=True)
        for yx in unique_yx[counts > 1]:
            mask = np.all(yx_arr == yx, axis=1)
            classes_at = cls_arr[mask]
            if len(set(classes_at)) > 1:
                descs_at = [label.get("description","") for label in meta_info.get("labels",[])
                            if any(np.array_equal(yx, p) for p in
                                (json.loads(label.get("pixels","[]")) if isinstance(label.get("pixels","[]"), str) else label.get("pixels",[])))]
                tqdm.tqdm.write(f"  [warn] conflicting labels at yx={yx.tolist()} classes={classes_at.tolist()} labels={descs_at}")
        return yx_arr, cls_arr


    CLASS_COLORS_RGB = np.array([[44, 160, 44], [214, 39, 40], [31, 119, 180]], dtype=np.float32)


    def scatter_dots(ax, ys, xs, labels_arr, title, show_legend=False):
        for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            mask = labels_arr == cls_idx
            if mask.any():
                ax.scatter(xs[mask], ys[mask], c=color, s=15, label=name,
                        linewidths=0, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if show_legend:
            ax.legend(loc="upper right", markerscale=2, fontsize=8)


    def scatter_with_errors(ax, ys, xs, gt_classes, pred, title, show_legend=False):
        """Draw predictions: correct→dot, wrong→X (pred color) + dot on top (GT color)."""
        correct = gt_classes == pred
        for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            m = correct & (gt_classes == cls_idx)
            if m.any():
                ax.scatter(xs[m], ys[m], c=color, s=15, label=name, linewidths=0, zorder=2)
        wrong = ~correct
        if wrong.any():
            for cls_idx, color in enumerate(CLASS_COLORS):   # X in predicted color
                m = wrong & (pred == cls_idx)
                if m.any():
                    ax.scatter(xs[m], ys[m], c=color, marker="x", s=80, linewidths=1.5, zorder=3)
            for cls_idx, color in enumerate(CLASS_COLORS):   # dot in GT color on top
                m = wrong & (gt_classes == cls_idx)
                if m.any():
                    ax.scatter(xs[m], ys[m], c=color, s=25, linewidths=0, zorder=4)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if show_legend:
            ax.legend(loc="upper right", markerscale=2, fontsize=7)


    def draw_vis(rgb_white, rgb_blue, ys, xs,
                gt_classes, pred_white_at_gt, pred_blue_at_gt, pred_blue_full,
                entry_name, out_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                                gridspec_kw={"wspace": 0.04})

        axes[0].imshow(rgb_white)
        scatter_dots(axes[0], ys, xs, gt_classes, "GT (white)", show_legend=True)

        axes[1].imshow(rgb_white)
        scatter_with_errors(axes[1], ys, xs, gt_classes, pred_white_at_gt,
                            "Pred (white)  ×=pred · •=GT")

        axes[2].imshow(rgb_blue)
        scatter_with_errors(axes[2], ys, xs, gt_classes, pred_blue_at_gt,
                            "Pred (blue)  ×=pred · •=GT")

        fig.suptitle(entry_name, fontsize=9, y=1.01)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    #---- iterate entries
    TOTOAL_ENTRY_NUM=1286
    entry_folders = sorted([f for f in os.listdir(PUB_DATA_ROOT) if re.match(r"entry_\d{4}", f)])
    if len(entry_folders) < TOTOAL_ENTRY_NUM:
        print("[WARNING] Evaluation on a subset {}/{}.".format(len(entry_folders),TOTOAL_ENTRY_NUM))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_gt, all_conf_white, all_conf_blue = [], [], []
    skipped_no_labels, skipped_illumination = 0, 0

    for entry_name in tqdm.tqdm(entry_folders, desc="Evaluating entries"):
        entry_root = PUB_DATA_ROOT + entry_name + "/"
        meta_path = entry_root + "meta_data.json"
        if not os.path.exists(meta_path):
            continue

        meta_info = json.load(open(meta_path, "r"))
        meta_info["labels"] = [l for l in meta_info.get("labels", [])
                            if l.get("description") != "coral_healthy_tip"] #"coral_healthy_tip" is unrealiable

        if meta_info.get("illumination") != "white":
            skipped_illumination += 1
            continue

        # check GT before any heavy IO
        gt_yx, gt_classes = parse_gt_pixels(meta_info)
        if gt_yx is None:
            skipped_no_labels += 1
            continue

        img_h = meta_info["image_height"]
        img_w = meta_info["image_width"]
        num_ch = meta_info["num_channels"]

        processed_path = entry_root + "processed_data/"
        reflectance_path = processed_path + "reflectance_cube.bin"
        if not os.path.exists(reflectance_path):
            continue

        reflectance_cube = (
            np.fromfile(reflectance_path, dtype=np.float32)
            .reshape((img_h * img_w, num_ch))[:, wavelength_indices]
        )
        with torch.no_grad():
            input_data = (
                torch.from_numpy(reflectance_cube).float().to(torch_device)
                .reshape((1, img_h, img_w, main_spectra.shape[0]))
                .permute(0, 3, 1, 2)
            )
            softmax_white = F.softmax(
                classifier(input_data).reshape((img_h, img_w, num_classes)), dim=2
            ).cpu().numpy()  # (H, W, 3)

        ys = np.clip(gt_yx[:, 0], 0, img_h - 1)
        xs = np.clip(gt_yx[:, 1], 0, img_w - 1)

        all_gt.append(gt_classes)
        all_conf_white.append(softmax_white[ys, xs])  # (N, 3)

        # --- blue sibling (always load) ---
        blue_name = meta_info.get("corresponding_entry", "")
        blue_root = PUB_DATA_ROOT + blue_name + "/"
        blue_processed = blue_root + "processed_data/"
        blue_cube_path = blue_processed + "reflectance_cube.bin"
        blue_meta_path = blue_root + "meta_data.json"

        softmax_blue = None
        if os.path.exists(blue_cube_path) and os.path.exists(blue_meta_path):
            blue_meta = json.load(open(blue_meta_path))
            blue_h, blue_w, blue_ch = blue_meta["image_height"], blue_meta["image_width"], blue_meta["num_channels"]
            blue_cube = (
                np.fromfile(blue_cube_path, dtype=np.float32)
                .reshape((blue_h * blue_w, blue_ch))[:, wavelength_indices]
            )
            with torch.no_grad():
                blue_input = (
                    torch.from_numpy(blue_cube).float().to(torch_device)
                    .reshape((1, blue_h, blue_w, main_spectra.shape[0]))
                    .permute(0, 3, 1, 2)
                )
                softmax_blue = F.softmax(
                    classifier(blue_input).reshape((blue_h, blue_w, num_classes)), dim=2
                ).cpu().numpy()  # (H, W, 3)
        else:
            tqdm.tqdm.write(f"  [warn] no blue cube for {entry_name}, filling NaN")
            softmax_blue = np.full((img_h, img_w, num_classes), np.nan, dtype=np.float32)
            blue_h, blue_w = img_h, img_w

        blue_ys = np.clip(ys, 0, blue_h - 1)
        blue_xs = np.clip(xs, 0, blue_w - 1)
        all_conf_blue.append(softmax_blue[blue_ys, blue_xs])  # (N, 3)

        # --- visualization (optional) ---
        if SAVE_VIS:
            pred_white_full = np.argmax(softmax_white, axis=2)
            pred_blue_full  = np.argmax(softmax_blue,  axis=2)

            rgb_path = processed_path + "reflectance_rgb.jpg"
            rgb_white = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB) \
                if os.path.exists(rgb_path) else np.zeros((img_h, img_w, 3), dtype=np.uint8)

            rgb_blue_path = blue_processed + "reflectance_rgb.jpg"
            rgb_blue = cv2.cvtColor(cv2.imread(rgb_blue_path), cv2.COLOR_BGR2RGB) \
                if os.path.exists(rgb_blue_path) else np.zeros((blue_h, blue_w, 3), dtype=np.uint8)

            draw_vis(rgb_white, rgb_blue, ys, xs,
                    gt_classes,
                    np.argmax(softmax_white[ys, xs], axis=1),
                    np.argmax(softmax_blue[blue_ys, blue_xs], axis=1),
                    pred_blue_full,
                    entry_name, OUTPUT_DIR + f"{entry_name}_vis.png")

    #---- aggregate
    all_gt         = np.concatenate(all_gt)
    all_conf_white = np.concatenate(all_conf_white)   # (N, 3)
    all_conf_blue  = np.concatenate(all_conf_blue)    # (N, 3)
    all_pred_white = np.argmax(all_conf_white, axis=1)
    all_pred_blue  = np.argmax(all_conf_blue,  axis=1)

    print(f"\nTotal labeled pixels  : {len(all_gt)}")
    print(f"Skipped (not white)   : {skipped_illumination}")
    print(f"Skipped (no labels)   : {skipped_no_labels}")

    def _report_block(gt, pred, label):
        n_correct = int(np.sum(pred == gt))
        acc = n_correct / len(gt)
        acc_line = f"Overall accuracy: {acc:.4f}  ({n_correct} / {len(gt)} correctly classified pixels)"
        report_str = classification_report(gt, pred, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4)
        report_str = "\n".join(l for l in report_str.splitlines() if not l.strip().startswith("accuracy")) + "\n"
        cm = confusion_matrix(gt, pred, labels=[0, 1, 2])
        cm_header = f"{'':>10}  " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
        cm_rows = [f"{CLASS_NAMES[i]:>10}  " + "  ".join(f"{v:>8}" for v in row) for i, row in enumerate(cm)]
        return acc_line, report_str, cm, cm_header, cm_rows

    acc_white, report_white, cm_white, cm_hdr, cm_rows_white = _report_block(all_gt, all_pred_white, "white")
    acc_blue,  report_blue,  cm_blue,  _,      cm_rows_blue  = _report_block(all_gt, all_pred_blue,  "blue")

    print(f"\n=== White light ===")
    print(acc_white)
    print("\n--- Classification Report ---")
    print(report_white)
    print("--- Confusion Matrix (rows=GT, cols=Pred) ---")
    print(cm_hdr)
    for row in cm_rows_white: print(row)

    print(f"\n=== Blue light (corresponding lighting condition) ===")
    print(acc_blue)
    print("\n--- Classification Report ---")
    print(report_blue)
    print("--- Confusion Matrix (rows=GT, cols=Pred) ---")
    print(cm_hdr)
    for row in cm_rows_blue: print(row)

    #---- save npz
    npz_path = OUTPUT_DIR + "eval_data.npz"
    np.savez(npz_path,
            gt=all_gt,
            conf_white=all_conf_white,
            conf_blue=all_conf_blue,
            pred_white=all_pred_white,
            pred_blue=all_pred_blue)
    print(f"Saved eval_data.npz → {npz_path}")

    #---- save text/json summary
    with open(OUTPUT_DIR + "eval_results.txt", "w") as f:
        f.write(f"Total labeled pixels  : {len(all_gt)}\n")
        f.write(f"Skipped (not white)   : {skipped_illumination}\n")
        f.write(f"Skipped (no labels)   : {skipped_no_labels}\n")
        for label, acc_line, report_str, cm, cm_rows in [
            ("White light",                          acc_white, report_white, cm_white, cm_rows_white),
            ("Blue light (corresponding condition)", acc_blue,  report_blue,  cm_blue,  cm_rows_blue),
        ]:
            f.write(f"\n=== {label} ===\n")
            f.write(acc_line + "\n\n")
            f.write("--- Classification Report ---\n" + report_str)
            f.write("\n--- Confusion Matrix (rows=GT, cols=Pred) ---\n")
            f.write("           " + "  ".join(f"{n:>8}" for n in CLASS_NAMES) + "\n")
            for row in cm_rows:
                f.write(row + "\n")

    precision_w, recall_w, f1_w, support_w = precision_recall_fscore_support(
        all_gt, all_pred_white, labels=[0, 1, 2], zero_division=0,
    )
    precision_b, recall_b, f1_b, _         = precision_recall_fscore_support(
        all_gt, all_pred_blue,  labels=[0, 1, 2], zero_division=0,
    )
    def _per_class_dict(precision, recall, f1, support):
        return {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
                "support":   int(support[i]),
            }
            for i in range(3)
        }
    results = {
        "total_pixels": int(len(all_gt)),
        "white_light": {
            "overall_accuracy": float(accuracy_score(all_gt, all_pred_white)),
            "per_class": _per_class_dict(precision_w, recall_w, f1_w, support_w),
            "confusion_matrix": cm_white.tolist(),
        },
        "blue_light": {
            "overall_accuracy": float(accuracy_score(all_gt, all_pred_blue)),
            "per_class": _per_class_dict(precision_b, recall_b, f1_b, support_w),
            "confusion_matrix": cm_blue.tolist(),
        },
    }
    with open(OUTPUT_DIR + "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}")
