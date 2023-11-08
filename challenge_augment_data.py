"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2

Script to create an augmented dataset.
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np
from scipy.ndimage import rotate
import torchvision.transforms as tortransformers
from imageio.v3 import imread, imwrite

import cv2

import rng_control


def Rotate(deg=20):
    """Return function to rotate image."""

    def _rotate(img):
        """Rotate a random integer amount in the range (-deg, deg) (inclusive).

        Keep the dimensions the same and fill any missing pixels with black.

        :img: H x W x C numpy array
        :returns: H x W x C numpy array
        """
        # TODO: implement _rotate(img)
        return rotate(input=img, angle=np.random.randint(-deg, deg), reshape=False)
        
    return _rotate



def Grayscale():
    """Return function to grayscale image."""

    def _grayscale(img):
        """Return 3-channel grayscale of image.

        Compute grayscale values by taking average across the three channels.

        Round to the nearest integer.

        :img: H x W x C numpy array
        :returns: H x W x C numpy array

        """
        # TODO: implement _grayscale(img)
        grayscale_img = np.mean(img, axis=2)
        grayscale_img = np.round(grayscale_img).astype(np.uint8)
        grayscale_img = np.stack([grayscale_img] * 3, axis=-1)
        return grayscale_img

    return _grayscale


def augment(filename, transforms, n=1, original=True):
    """Augment image at filename.

    :filename: name of image to be augmented
    :transforms: List of image transformations
    :n: number of augmented images to save
    :original: whether to include the original images in the augmented dataset or not
    :returns: a list of augmented images, where the first image is the original

    """
    print(f"Augmenting {filename}")
    img = imread(filename)
    res = [img] if original else []
    for i in range(n):
        new = img
        for transform in transforms:
            new = transform(new)
        res.append(new)
    return res

transform_shape = tortransformers.Compose([
    tortransformers.RandomRotation(degrees=(-7, 7)),
    # tortransformers.RandomResizedCrop(
    #     (64, 64), scale=(0.7, 1), ratio=(0.5, 2)),
    tortransformers.RandomHorizontalFlip(),
    # tortransformers.ToTensor(),
])

transformer_color = tortransformers.ColorJitter(
  brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)

def main(args):
    """Create augmented dataset."""
    
    use_feature_selection = input("do you want to use feature selection? y/n\n")
    
    reader = csv.DictReader(open(args.input, "r"), delimiter=",")
    writer = csv.DictWriter(
        open(f"{args.datadir}/augmented_landmarks.csv", "w"),
        fieldnames=["filename", "semantic_label", "partition", "numeric_label", "task"],
    )
    augment_partitions = set(args.partitions)

    # TODO: change `augmentations` to specify which augmentations to apply
    # augmentations = [Grayscale(), Rotate()]
    # augmentations = [Grayscale(), tortransformers.ToPILImage(), transformer_color]
    augmentations = [tortransformers.ToPILImage(), transform_shape, transformer_color]
    # augmentations = [Rotate()]

    writer.writeheader()
    os.makedirs(f"{args.datadir}/challenge_augmented/", exist_ok=True)
    for f in glob.glob(f"{args.datadir}/challenge_augmented/*"):
        print(f"Deleting {f}")
        os.remove(f)
    for row in reader:
        if row["partition"] not in augment_partitions:
            imwrite(
                f"{args.datadir}/challenge_augmented/{row['filename']}",
                imread(f"{args.datadir}/images/{row['filename']}"),
            )
            writer.writerow(row)
            continue
        imgs = augment(
            f"{args.datadir}/images/{row['filename']}",
            augmentations,
            n=1,
            original=True,  # TODO: change to False to exclude original image.
        )
        for i, img in enumerate(imgs):
            fname = f"{row['filename'][:-4]}_aug_{i}.png"
          
            imwrite(f"{args.datadir}/challenge_augmented/{fname}", img)
            writer.writerow(
                {
                    "filename": fname,
                    "semantic_label": row["semantic_label"],
                    "partition": row["partition"],
                    "numeric_label": row["numeric_label"],
                    "task": row["task"],
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("datadir", help="Data directory", default="./data/")
    parser.add_argument(
        "-p",
        "--partitions",
        nargs="+",
        help="Partitions (train|val|test|challenge|none)+ to apply augmentations to. Defaults to train",
        default=["train"],
    )
    main(parser.parse_args(sys.argv[1:]))
