"""
Load a trained UrbanSoundAI model and predict the sound class for a given .wav file.
Usage: python predict.py path/to/audio.wav [--model path/to/model.pt]
"""
import os
import argparse
import torch
import numpy as np

from src.config import MODELS_DIR, DEFAULT_MODEL_NAME
from src.dataset import audio_to_mel_spectrogram
from src.model import get_model


def load_checkpoint(model_path):
    """Load saved checkpoint; returns model and class list."""
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]
    model = get_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def predict_wav(wav_path, model, classes, device="cpu"):
    """
    Run prediction on one WAV file.
    Returns predicted class name and optional probabilities.
    """
    spec = audio_to_mel_spectrogram(wav_path)
    # (1, n_mels, time) as batch
    x = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
    return classes[pred_idx], probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict sound class from a WAV file")
    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to the .wav audio file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME),
        help="Path to trained model .pt file",
    )
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="Print probability for each class",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav_path):
        print(f"Error: File not found: {args.wav_path}")
        return
    if not os.path.isfile(args.model):
        print(f"Error: Model not found: {args.model}. Run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_checkpoint(args.model)
    model = model.to(device)

    pred_class, probs = predict_wav(args.wav_path, model, classes, device)
    print(f"Predicted class: {pred_class}")
    if args.show_probs:
        print("Probabilities:")
        for c, p in zip(classes, probs):
            print(f"  {c}: {p:.3f}")


if __name__ == "__main__":
    main()
