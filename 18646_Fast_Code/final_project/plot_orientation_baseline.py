import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_orientation_baseline.py results.json")
    sys.exit(1)

json_path = sys.argv[1]
output_dir = os.path.dirname(os.path.abspath(json_path))

with open(json_path) as f:
    data = json.load(f)

runtimes_ms = np.array(data["all_results_ms"])
input_kps   = np.array(data["all_input_kp_counts"])
image_areas = np.array(data["all_image_areas"])

# ── bucket images by standard AFHQ sizes ─────────────────────────────────
buckets = [
    ("128x128",   0,        128*128 + 1000),
    ("256x256",   128*128,  256*256 + 5000),
    ("512x512",   256*256,  512*512 + 10000),
    ("768x768",   512*512,  768*768 + 20000),
    ("1024x1024", 768*768,  np.inf),
]

labels, means, stds, median_kps = [], [], [], []
for label, lo, hi in buckets:
    mask = (image_areas > lo) & (image_areas <= hi)
    if mask.sum() == 0:
        continue
    labels.append(label)
    means.append(runtimes_ms[mask].mean())
    stds.append(runtimes_ms[mask].std())
    median_kps.append(int(np.median(input_kps[mask])))

# ── figure ────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("PythonSIFT Baseline (AFHQ) — Orientation Assignment",
             fontsize=13, fontweight="bold")

ORIENT_COLOR = "#C0392B"

# ── Fig 1: bar chart by image size ───────────────────────────────────────
x = np.arange(len(labels))
ax1.bar(x, means, width=0.5, color=ORIENT_COLOR, alpha=0.88,
        yerr=stds, capsize=5,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
        label="Orientation assignment (step 3)")

for i, (m, s, kps) in enumerate(zip(means, stds, median_kps)):
    ax1.text(i, m + s + max(means) * 0.01,
             f"{kps:,} kps",
             ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Image Size")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Fig 1: Orientation assignment latency by image size\n(avg over AFHQ images)")
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
ax1.set_ylim(bottom=0)

# ── Fig 2: scatter + linear fit, keypoints vs latency (ms) ───────────────
# remove top 1% outliers for a clean fit line
mask = runtimes_ms < np.percentile(runtimes_ms, 99)
slope, intercept = np.polyfit(input_kps[mask], runtimes_ms[mask], 1)
fit_x = np.linspace(0, input_kps.max(), 400)
fit_y = slope * fit_x + intercept

ax2.scatter(input_kps, runtimes_ms, color=ORIENT_COLOR, s=30, zorder=3)
ax2.plot(fit_x, fit_y, color="#888888", linewidth=1.5, linestyle="--",
         label=f"Linear fit: {slope:.2f} ms/kp + {intercept:.1f} ms",
         zorder=2)

ax2.set_xlabel("Number of Keypoints")
ax2.set_ylabel("Orientation Assignment Latency (ms)")
ax2.set_title("Fig 2: Orientation latency vs keypoint count\n(all AFHQ images, orientation only)")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.set_ylim(bottom=0)

plt.tight_layout()
output_path = os.path.join(output_dir, "orientation_baseline_plots.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
print(f"Linear fit: {slope:.2f} ms/keypoint + {intercept:.1f} ms")