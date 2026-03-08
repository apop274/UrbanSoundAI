# UrbanSoundAI

A machine learning project that trains a CNN to classify environmental sounds from the **UrbanSound8K** dataset using mel spectrograms. This version uses four classes: **siren**, **dog_bark**, **drilling**, and **engine_idling**.

## Dataset

The extracted dataset lives under `data/UrbanSound8K/`: **audio** is in `audio/fold1` … `audio/fold10`, and **metadata** is in `metadata/UrbanSound8K.csv`. The CSV lists every clip with columns such as slice file name, fold, start/end time, and **class** (e.g. siren, dog_bark, drilling, engine_idling)—so you can see all the data and labels in one place.

## Screenshots



### Sample audio: 62048-3-0-3 (fold 8)

Example clip from the dataset: `data/UrbanSound8K/audio/fold8/62048-3-0-3.wav`.
<img width="1029" height="48" alt="image" src="https://github.com/user-attachments/assets/d963da41-c493-4f88-aeaf-7d38413c9b0f" />





### Matplotlib spectrogram output

<img width="997" height="430" alt="image" src="https://github.com/user-attachments/assets/23a3882c-40a6-4db6-9a3f-44dd13ebb9d5" />



---

## Project structure

```
UrbanSoundAI/
    assets/                   # Screenshots for README (drag & drop here)
    data/
        UrbanSound8K/          # Dataset (audio/fold1..fold10, metadata/UrbanSound8K.csv)
    extract_dataset.py        # Extract dataset from .tar.gz / .gz download
    outputs/
        models/                # Saved trained models (.pt)
        figures/               # Saved spectrogram plots
    src/
        __init__.py
        config.py              # Paths, classes, spectrogram & training settings
        dataset.py             # Load CSV, filter classes, mel spectrograms, PyTorch Dataset
        model.py               # CNN definition
    train.py                   # Train CNN, print accuracy, save model
    predict.py                 # Predict class for a .wav file
    visualize_sample.py        # Display mel spectrogram of a WAV file
    requirements.txt
    README.md
```

## Setup

1. **Python**: Use Python 3.8 or newer.

2. **Dataset** (for training):
   - If you have the **downloaded .gz / .tar.gz archive**, extract it into the project:
     ```bash
     python extract_dataset.py path/to/UrbanSound8K.tar.gz
     ```
     (Use the path where your download is, e.g. `C:\Users\You\Downloads\UrbanSound8K.tar.gz`.)
   - Or place an already-extracted dataset under:
     - `UrbanSoundAI/data/UrbanSound8K/`
     - With `audio/fold1` … `audio/fold10` and `metadata/UrbanSound8K.csv`.

3. **Install dependencies**:
   ```bash
   cd UrbanSoundAI
   pip install -r requirements.txt
   ```

## Usage

### Train the model

```bash
python train.py
```

Optional arguments:
- `--model-name urbansound_cnn.pt` – output filename (default: `urbansound_cnn.pt`)
- `--epochs 25` – number of epochs
- `--batch-size 32` – batch size

The script loads the CSV, filters to the four classes, builds mel spectrograms, splits into train/test, trains the CNN, prints training progress and final test accuracy, and saves the model to `outputs/models/`.

### Predict on a WAV file

```bash
python predict.py path/to/audio.wav
```

Options:
- `--model path/to/model.pt` – path to saved model (default: `outputs/models/urbansound_cnn.pt`)
- `--show-probs` – print probability for each of the four classes

### Visualize a spectrogram

```bash
python visualize_sample.py path/to/audio.wav
```

Options:
- `--title "My title"` – plot title
- `--save path/to/figure.png` – save figure (default: `outputs/figures/sample_spectrogram.png`)
- `--no-show` – only save, do not open the plot window

## Tech stack

- **PyTorch** – CNN and training
- **librosa** – load audio and compute mel spectrograms
- **matplotlib** – spectrogram visualization
- **pandas** – read UrbanSound8K metadata CSV
- **scikit-learn** – train/test split

## Model

A small CNN takes mel spectrograms of shape `(1, n_mels, time_steps)` (128 mel bins, 173 time steps). It uses three conv blocks (with BatchNorm, ReLU, MaxPool, Dropout), global average pooling, and two fully connected layers to output four class logits.

## License

UrbanSound8K has its own license terms; ensure you comply with them when using the dataset.
