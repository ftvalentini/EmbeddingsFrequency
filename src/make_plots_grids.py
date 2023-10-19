
from pathlib import Path

import cv2 as cv


def main():

    ### Cosine unshuffled
    files = [
        "results/plots/heatmap-SGNS-cosine.png",
        "results/plots/heatmap-FastText-cosine.png",
        "results/plots/heatmap-GloVe-cosine.png",
    ]
    nombre = "heatmap-cosine"
    imgs = []
    for f in files:
        basename = Path(f).stem
        img = cv.imread(f)
        imgs.append(img)

    imgs = cv.hconcat(imgs)
    cv.imwrite(f"results/plots/grid_{nombre}.png", imgs)

    ### Cosine shuffled
    files = [
        "results/plots/heatmap-SGNS-cosine_s0.png",
        "results/plots/heatmap-FastText-cosine_s0.png",
        "results/plots/heatmap-GloVe-cosine_s0.png",
    ]
    nombre = "heatmap-cosine_shuffled"
    imgs = []
    for f in files:
        basename = Path(f).stem
        img = cv.imread(f)
        imgs.append(img)
    imgs = cv.hconcat(imgs)
    cv.imwrite(f"results/plots/grid_{nombre}.png", imgs)

    ### Euclidean distance unshuffled
    files = [
        "results/plots/heatmap-SGNS-negative_distance.png",
        "results/plots/heatmap-FastText-negative_distance.png",
        "results/plots/heatmap-GloVe-negative_distance.png",
    ]
    nombre = "heatmap-euclidian"
    imgs = []
    for f in files:
        basename = Path(f).stem
        img = cv.imread(f)
        imgs.append(img)

    imgs = cv.hconcat(imgs)
    cv.imwrite(f"results/plots/grid_{nombre}.png", imgs)

    ### Euclidean distance shuffled
    files = [
        "results/plots/heatmap-SGNS-negative_distance_s0.png",
        "results/plots/heatmap-FastText-negative_distance_s0.png",
        "results/plots/heatmap-GloVe-negative_distance_s0.png",
    ]
    nombre = "heatmap-euclidian_shuffled"
    imgs = []
    for f in files:
        basename = Path(f).stem
        img = cv.imread(f)
        imgs.append(img)
    imgs = cv.hconcat(imgs)
    cv.imwrite(f"results/plots/grid_{nombre}.png", imgs)

    # ### PCA scatter+centroids (normalized)
    files = [
        "results/plots/grid_pca-normalized_FastText.png",
        "results/plots/grid_pca-normalized_GloVe.png",
    ]
    nombre = "grid_pca-normalized-rest"
    imgs = []
    for f in files:
        basename = Path(f).stem
        img = cv.imread(f)
        imgs.append(img)
    imgs = cv.hconcat(imgs)
    cv.imwrite(f"results/plots/{nombre}.png", imgs)


if __name__ == "__main__":
    main()
