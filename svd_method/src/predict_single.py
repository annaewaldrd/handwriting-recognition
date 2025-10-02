"""
Predict a digit using a trained SVD model.

- Loads a saved model (.npz)
- Preprocesses a single input image
- Outputs predicted digit and reconstruction error
"""

import argparse
import os
from test import load_model, classify_vector
from utils import preprocess_image

def main(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Load trained SVD model
    model = load_model(args.model)

    # Preprocess input image
    x = preprocess_image(args.image, size=(args.size, args.size), invert=args.invert)
    
    # Predict digit by finding minimal reconstruction error
    pred, err = classify_vector(x, model)
    print(f"Predicted digit: {pred} (reconstruction error {err:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/svd_model.npz")
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--size", type=int, default=8, help="Resize image (8 for sklearn digits, 28 for MNIST-like)")
    parser.add_argument("--invert", action="store_true", help="Invert colors if needed")
    args = parser.parse_args()
    main(args)
