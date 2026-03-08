"""
Extract the UrbanSound8K dataset from a .tar.gz or .gz archive into the project.
Use this if you have the downloaded archive (e.g. UrbanSound8K.tar.gz) and have not
extracted it yet.

Usage:
    python extract_dataset.py path/to/UrbanSound8K.tar.gz
    python extract_dataset.py path/to/archive.gz

The script will create/use data/UrbanSound8K/ with audio/ (fold1..fold10) and metadata/ inside.
"""
import os
import sys
import tarfile
import gzip
import shutil
import tempfile

# Project data dir where we want: data/UrbanSound8K/audio/ and data/UrbanSound8K/metadata/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "UrbanSound8K")


def extract_tar_gz(archive_path, dest_dir):
    """Extract a .tar.gz so that dest_dir contains audio/ and metadata/."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            print("Archive is empty.")
            return False
        names = tar.getnames()
        first = names[0]
        total = len(members)
        print(f"Found {total} files. Extracting (this can take 5–15 minutes for a large archive)...")
        # Extract one by one so we can show progress
        for i, member in enumerate(members):
            tar.extract(member, path=dest_dir)
            if (i + 1) % 500 == 0 or i == 0 or i == total - 1:
                print(f"  ... {i + 1}/{total} files")
        # Common case: archive root is "UrbanSound8K" or "UrbanSound8K/"
        if first.startswith("UrbanSound8K"):
            inner = os.path.join(dest_dir, "UrbanSound8K")
            if os.path.isdir(inner):
                for item in os.listdir(inner):
                    shutil.move(os.path.join(inner, item), os.path.join(dest_dir, item))
                os.rmdir(inner)
            return True
        # Single top-level folder that contains audio & metadata
        top_items = os.listdir(dest_dir)
        if len(top_items) == 1:
            only = os.path.join(dest_dir, top_items[0])
            if os.path.isdir(only):
                for item in os.listdir(only):
                    shutil.move(os.path.join(only, item), os.path.join(dest_dir, item))
                os.rmdir(only)
        return True


def extract_plain_gz(archive_path, dest_dir):
    """Handle a plain .gz (single file). UrbanSound8K is usually .tar.gz, but we support this."""
    os.makedirs(dest_dir, exist_ok=True)
    # Assume it's a gzipped tar
    out_path = os.path.join(dest_dir, "archive.tar")
    with gzip.open(archive_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    try:
        with tarfile.open(out_path, "r") as tar:
            names = tar.getnames()
            first = names[0] if names else ""
            tar.extractall(path=dest_dir)
            inner = os.path.join(dest_dir, "UrbanSound8K")
            if os.path.isdir(inner):
                for item in os.listdir(inner):
                    shutil.move(os.path.join(inner, item), os.path.join(dest_dir, item))
                os.rmdir(inner)
    finally:
        if os.path.isfile(out_path):
            os.remove(out_path)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_dataset.py <path_to_archive.gz_or_tar.gz>")
        print("Example: python extract_dataset.py C:\\Downloads\\UrbanSound8K.tar.gz")
        sys.exit(1)

    archive_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(archive_path):
        print(f"Error: File not found: {archive_path}")
        sys.exit(1)

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        extract_fn = extract_tar_gz
    elif archive_path.endswith(".gz"):
        extract_fn = extract_plain_gz
    else:
        print("Expected a .gz or .tar.gz file.")
        sys.exit(1)

    print(f"Extracting to {TARGET_DIR} ...")
    print("(UrbanSound8K is large — please wait, do not press Ctrl+C.)")
    os.makedirs(TARGET_DIR, exist_ok=True)
    try:
        extract_fn(archive_path, TARGET_DIR)
    except Exception as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)

    # Verify expected structure
    audio_dir = os.path.join(TARGET_DIR, "audio")
    csv_path = os.path.join(TARGET_DIR, "metadata", "UrbanSound8K.csv")
    if os.path.isdir(audio_dir) and os.path.isfile(csv_path):
        print("Done. Dataset is ready at data/UrbanSound8K/ (audio/ and metadata/).")
    else:
        print("Extraction finished. Please check that data/UrbanSound8K/ contains audio/ and metadata/UrbanSound8K.csv.")


if __name__ == "__main__":
    main()
