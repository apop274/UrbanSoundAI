"""
Load one audio file, compute its mel spectrogram with librosa,
and display it with matplotlib. Optionally save the figure.
"""
import os
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.config import SR, N_FFT, HOP_LENGTH, N_MELS, FIGURES_DIR


def compute_mel_spectrogram(file_path):
    """Load audio and return mel spectrogram in dB (same as in dataset)."""
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, y, SR


def plot_spectrogram(mel_db, sr=SR, hop_length=HOP_LENGTH, title="Mel spectrogram", save_path=None, show=True):
    """Display mel spectrogram and optionally save to file."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize mel spectrogram of a WAV file")
    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to the .wav audio file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mel spectrogram",
        help="Title for the plot",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save figure to this path (default: outputs/figures/sample_spectrogram.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (only save if --save is set)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav_path):
        print(f"Error: File not found: {args.wav_path}")
        return

    mel_db, _, sr = compute_mel_spectrogram(args.wav_path)
    title = args.title or os.path.basename(args.wav_path)
    # Only save to disk when user asks: --save [path] or --no-show (saves to default path)
    if args.save is not None:
        save_path = args.save
    elif args.no_show:
        save_path = os.path.join(FIGURES_DIR, "sample_spectrogram.png")
    else:
        save_path = None

    plot_spectrogram(
        mel_db, sr=sr, title=title, save_path=save_path, show=not args.no_show
    )
    if args.no_show:
        plt.close()


if __name__ == "__main__":
    main()
