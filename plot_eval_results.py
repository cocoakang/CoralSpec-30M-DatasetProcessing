import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coral masks for spectral dataset using classifier and SAM.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root directory")
    args = parser.parse_args()

    PUB_DATA_ROOT=args.data_path
    OUTPUT_DIR = PUB_DATA_ROOT + "eval_classifier/"

    NPZ_PATH   = OUTPUT_DIR+"eval_data.npz"

    CLASS_NAMES = ["healthy", "sick", "others"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    data = np.load(NPZ_PATH)
    gt         = data["gt"]           # (N,)
    conf_white = data["conf_white"]   # (N, 3)
    conf_blue  = data["conf_blue"]    # (N, 3)

    # one-vs-rest binary label for each class
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8))

    for col, cls_idx in enumerate(range(3)):
        gt_binary = (gt == cls_idx).astype(int)

        for conf, label, color, ls in [
            (conf_white, "white", "#1f77b4", "-"),
            (conf_blue,  "blue",  "#ff7f0e", "--"),
        ]:
            score = conf[:, cls_idx]
            valid = ~np.isnan(score)

            # ROC
            fpr, tpr, _ = roc_curve(gt_binary[valid], score[valid])
            roc_auc = auc(fpr, tpr)
            axes[0, col].plot(fpr, tpr, color=color, ls=ls, lw=1.4,
                            label=f"{label} AUC={roc_auc:.3f}")

            # PR
            prec, rec, _ = precision_recall_curve(gt_binary[valid], score[valid])
            ap = average_precision_score(gt_binary[valid], score[valid])
            axes[1, col].plot(rec, prec, color=color, ls=ls, lw=1.4,
                            label=f"{label} AP={ap:.3f}")

        # ROC baseline
        axes[0, col].plot([0, 1], [0, 1], color="#e15759", ls=":", lw=1.0)
        axes[0, col].set_title(f"ROC — {CLASS_NAMES[cls_idx]}", fontsize=8)
        axes[0, col].set_xlabel("FPR")
        axes[0, col].set_ylabel("TPR")
        axes[0, col].legend(loc="lower right")

        # PR baseline
        base_rate = gt_binary.mean()
        axes[1, col].axhline(base_rate, color="#e15759", ls=":", lw=1.0)
        axes[1, col].set_title(f"PR — {CLASS_NAMES[cls_idx]}", fontsize=8)
        axes[1, col].set_xlabel("Recall")
        axes[1, col].set_ylabel("Precision")
        axes[1, col].legend(loc="upper right")

    # hide repeated y-labels on middle/right columns
    for row in range(2):
        for col in [1, 2]:
            axes[row, col].tick_params(labelleft=False)
            axes[row, col].set_ylabel("")

    fig.tight_layout()
    out_path = OUTPUT_DIR + "roc_pr.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
