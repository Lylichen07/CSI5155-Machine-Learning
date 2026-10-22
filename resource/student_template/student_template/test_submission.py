"""Check your Part 3 model on 20 HF test images and zip Part1, Part2 and Part3 for submission.

On a T4 GPU it also checks predict() is fast enough: 5000 images in under 10 minutes, measured on
250 images or for 1 minute, whichever comes first.
Pass --speed-test to run that check on other hardware too (the limit is meant for a T4).

Install: python -m pip install -r requirements.txt
Run: python test_submission.py --student-number 12345
Dataset: Shanmuk4622/ai-image-detection-dataset (default configuration, test split).
"""
import argparse
import os
import subprocess
import sys
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from datasets import load_dataset
import torch

DATASET = "Shanmuk4622/ai-image-detection-dataset"
PARTS = ("Part1", "Part2", "Part3")
INTERFACE = ("Part3/inference.py", "Part3/weights.pt", "requirements.txt")
IGNORED = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
NUM_IMAGES = 20
SPEED_IMAGES = 5000   # predict must handle SPEED_IMAGES images in under SPEED_LIMIT seconds on a T4
SPEED_LIMIT = 10 * 60
SPEED_SAMPLE = 250    # the speed test stops after SPEED_SAMPLE images or SPEED_DURATION seconds, whichever is first
SPEED_DURATION = 60
SPEED_CHUNK = 25      # images per predict call; the time limit is checked between calls

# Runs in a separate isolated interpreter (-I) inside the unzipped submission.
# Checks predict()'s output here too, in case its own assertions were edited away.
# With a speed folder of image chunks and a duration, it then calls predict on each chunk until
# they are done or the duration has passed, and prints the image count and elapsed seconds.
WORKER = """
import os
import sys
import time
import torch
sys.path.insert(0, 'Part3')  # -I removes the working directory from sys.path
from inference import load_model, predict
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load_model(device)
labels = predict(model, sys.argv[1], 'train', device)
assert type(labels) is list, "predict must return a list of labels"
assert len(labels) == len(os.listdir(sys.argv[1])), "predict must return one label per image"
assert all(type(label) is int and label in (0, 1) for label in labels), "Labels must be Python integers: 0=real, 1=AI"
if len(sys.argv) > 2:
    count, start = 0, time.perf_counter()
    for chunk in sorted(os.listdir(sys.argv[2])):
        count += len(predict(model, os.path.join(sys.argv[2], chunk), 'train', device))
        if time.perf_counter() - start >= float(sys.argv[3]):
            break
    print(count, time.perf_counter() - start, device)
"""


def is_t4():
    return torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)


def save_test_images(folder, count):
    """Write the first count HF test images to a flat image folder without labels, as the evaluator passes to predict."""
    folder.mkdir()
    rows = load_dataset(DATASET, name="default", split="test", streaming=True)
    for i, row in enumerate(islice(rows, count)):
        row["image"].convert("RGB").save(folder / f"{i:06d}.png")
    return len(list(folder.iterdir()))


def chunk_folder(folder, size):
    """Move the images in folder into numbered subfolders of size images each."""
    files = sorted(folder.iterdir())
    for i in range(0, len(files), size):
        chunk = folder / f"chunk{i // size:03d}"
        chunk.mkdir()
        for file in files[i:i + size]:
            file.rename(chunk / file.name)


def project_files(folder):
    """Every file under Part1, Part2 and Part3, plus requirements.txt, as (path, archive name)."""
    files = [(folder / "requirements.txt", "requirements.txt")]
    for part in PARTS:
        for path in sorted((folder / part).rglob("*")):
            relative = path.relative_to(folder)
            if path.is_file() and not IGNORED & set(relative.parts) and path.suffix != ".zip":
                files.append((path, relative.as_posix()))
    return files


def package(folder, student_number, speed_test=False):
    folder = Path(folder).resolve()
    if not student_number.isdigit():
        raise SystemExit("FAIL: student number must contain only digits")
    missing = [part for part in PARTS if not (folder / part).is_dir()]
    missing += [name for name in INTERFACE if not (folder / name).is_file()]
    if missing:
        raise SystemExit(f"FAIL: missing: {', '.join(missing)}")

    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        archive = temporary / f"submission_{student_number}.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
            for path, name in project_files(folder):
                bundle.write(path, name)

        # Test the zip contents, not the working folder.
        extracted = temporary / "extracted"
        with ZipFile(archive) as bundle:
            bundle.extractall(extracted)

        images = temporary / "images"
        saved = save_test_images(images, NUM_IMAGES)
        if saved != NUM_IMAGES:
            raise SystemExit(f"FAIL: expected {NUM_IMAGES} test images, got {saved}")

        command = [sys.executable, "-I", "-B", "-c", WORKER, str(images)]
        speed_test = speed_test or is_t4()
        if speed_test:
            speed_images = temporary / "speed_images"
            save_test_images(speed_images, SPEED_SAMPLE)
            chunk_folder(speed_images, SPEED_CHUNK)
            command += [str(speed_images), str(SPEED_DURATION)]

        # EVALUATE=True stops a script exported from inference.ipynb from training on import.
        process = subprocess.run(command, cwd=extracted, capture_output=True, text=True,
                                 env={**os.environ, "EVALUATE": "True"})
        if process.returncode:
            raise SystemExit(f"FAIL: Part3/inference.py raised an error:\n{process.stderr}")

        if speed_test:
            count, elapsed, device = process.stdout.split()[-3:]
            count, elapsed = int(count), float(elapsed)
            projected = SPEED_IMAGES * elapsed / count
            print(f"{count} images in {elapsed:.1f}s on {device}: {count / elapsed:.1f} images/s, "
                  f"projected {projected:.0f}s for {SPEED_IMAGES} images (limit {SPEED_LIMIT}s)")
            if projected >= SPEED_LIMIT:
                raise SystemExit(f"FAIL: too slow: {SPEED_IMAGES} images would take {projected:.0f}s, "
                                 f"limit is {SPEED_LIMIT}s")

        # Only replace an existing submission once the new one passed.
        output = folder / archive.name
        output.write_bytes(archive.read_bytes())

    speed = f" and is fast enough for {SPEED_IMAGES} images" if speed_test else ""
    print(f"PASS: Part 3 model loads and predicts valid labels on {NUM_IMAGES} HF test images{speed}.\nSubmit {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-number", required=True)
    parser.add_argument("--speed-test", action="store_true",
                        help="run the speed check even without a T4 GPU (it always runs on a T4)")
    args = parser.parse_args()
    package(Path(__file__).resolve().parent, args.student_number, args.speed_test)
