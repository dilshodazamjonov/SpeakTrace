"""Validate the raw Speak & Improve data: archives and the extracted FLAC tree.

Run from the repository root, after data_download.py has finished:

    python ml/src/validate_raw_data.py

Checks:
    1. No archive is missing (every zip from data_links is on disk).
    2. Every ZIP opens successfully.
    3. No duplicate filenames inside an archive (would overwrite on flat extract).
    4. Extracted FLAC count matches the archive contents, per zip.
    5. No duplicate relative paths or filenames in the extracted tree.
    6. Every FLAC can be opened.
    7. Duration is positive.
    8. Sample rate and channel count are as expected.
"""

import os
import sys
from collections import Counter
from zipfile import BadZipFile, ZipFile

import soundfile as sf
from tqdm import tqdm

from .data_download import AUDIO_EXTENSION, data_links

ARCHIVE_DIR = 'ml/data/raw/archives/'
DATA_DIR = 'ml/data/raw/data/'

# Adjust if the corpus specs differ; mismatches are reported with the observed values.
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


def zip_subdir(zip_name: str) -> str:
    # Same naming rule as data_download.zip_transfer:
    # "data.flac.dev.01.zip" -> "dev01", "data.flac.train.04.P1.zip" -> "train04-P1"
    name = zip_name.removeprefix('data.flac.').removesuffix('.zip')
    return name.replace('.', '', 1).replace('.', '-')


def zip_split(zip_name: str) -> str | None:
    for split in ('dev', 'train', 'eval'):
        if split in zip_name:
            return split
    return None


def check_archives_present(expected_zips: list[str]) -> list[str]:
    errors = []

    for zip_name in expected_zips:
        if not os.path.isfile(os.path.join(ARCHIVE_DIR, zip_name)):
            errors.append(f"Missing archive: {zip_name}")

    return errors


def check_zip_archives(expected_zips: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    errors = []
    zip_flacs: dict[str, list[str]] = {}

    for zip_name in tqdm(expected_zips, desc="Opening archives", unit="zip"):
        path = os.path.join(ARCHIVE_DIR, zip_name)

        if not os.path.isfile(path):
            continue  # already reported by check_archives_present

        try:
            with ZipFile(path, 'r') as zip_ref:
                members = [
                    info.filename for info in zip_ref.infolist()
                    if info.filename.lower().endswith(AUDIO_EXTENSION)
                ]
        except (BadZipFile, OSError) as exc:
            errors.append(f"Corrupt archive, cannot open: {zip_name} ({exc})")
            continue

        zip_flacs[zip_name] = members

        name_counts = Counter(os.path.basename(m) for m in members)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            errors.append(
                f"{zip_name}: {len(duplicates)} duplicate filename(s) inside the archive, "
                f"flat extraction overwrites them (e.g. {duplicates[:3]})"
            )

    return errors, zip_flacs


def check_extracted_counts(zip_flacs: dict[str, list[str]]) -> list[str]:
    errors = []

    for zip_name, members in zip_flacs.items():
        split = zip_split(zip_name)
        if split is None:
            errors.append(f"{zip_name}: cannot determine split (dev/train/eval)")
            continue

        extract_dir = os.path.join(DATA_DIR, split, zip_subdir(zip_name))

        if not os.path.isdir(extract_dir):
            errors.append(f"{zip_name}: extraction folder missing: {extract_dir}")
            continue

        extracted = [
            f for f in os.listdir(extract_dir)
            if f.lower().endswith(AUDIO_EXTENSION)
        ]

        if len(extracted) != len(members):
            errors.append(
                f"{zip_name}: archive has {len(members)} FLAC files "
                f"but {len(extracted)} extracted in {extract_dir}"
            )

    return errors


def collect_flac_paths() -> list[str]:
    flac_paths = []

    for root, _dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            if file_name.lower().endswith(AUDIO_EXTENSION):
                flac_paths.append(os.path.join(root, file_name))

    return flac_paths


def check_duplicates(flac_paths: list[str]) -> list[str]:
    errors = []

    rel_paths = Counter(
        os.path.relpath(path, DATA_DIR).lower() for path in flac_paths
    )
    for rel_path, count in rel_paths.items():
        if count > 1:
            errors.append(f"Duplicate relative path ({count}x): {rel_path}")

    file_names = Counter(os.path.basename(path) for path in flac_paths)
    for file_name, count in file_names.items():
        if count > 1:
            errors.append(f"Duplicate filename across folders ({count}x): {file_name}")

    return errors


def check_flac_files(flac_paths: list[str]) -> list[str]:
    errors = []
    sample_rates: Counter = Counter()
    channel_counts: Counter = Counter()

    for path in tqdm(flac_paths, desc="Checking FLAC files", unit="file"):
        try:
            info = sf.info(path)
        except (RuntimeError, OSError) as exc:
            errors.append(f"Cannot open: {path} ({exc})")
            continue

        sample_rates[info.samplerate] += 1
        channel_counts[info.channels] += 1

        if info.duration <= 0:
            errors.append(f"Non-positive duration ({info.duration}s): {path}")
        if info.samplerate != EXPECTED_SAMPLE_RATE:
            errors.append(f"Sample rate {info.samplerate}, expected {EXPECTED_SAMPLE_RATE}: {path}")
        if info.channels != EXPECTED_CHANNELS:
            errors.append(f"{info.channels} channel(s), expected {EXPECTED_CHANNELS}: {path}")

    print(f"Observed sample rates: {dict(sample_rates)}")
    print(f"Observed channel counts: {dict(channel_counts)}")

    return errors


def main() -> int:
    expected_zips = [link.split('/')[-1] for link in data_links]
    all_errors: list[str] = []

    print("=== 1. Archive presence ===")
    all_errors += check_archives_present(expected_zips)

    print("=== 2. Archive integrity ===")
    zip_errors, zip_flacs = check_zip_archives(expected_zips)
    all_errors += zip_errors

    print("=== 3. Extracted counts vs archive contents ===")
    all_errors += check_extracted_counts(zip_flacs)

    print("=== 4. Duplicates in extracted tree ===")
    flac_paths = collect_flac_paths()
    print(f"Found {len(flac_paths)} FLAC files under {DATA_DIR}")
    all_errors += check_duplicates(flac_paths)

    print("=== 5. FLAC readability, duration, sample rate, channels ===")
    all_errors += check_flac_files(flac_paths)

    print()
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} problem(s) found")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("VALIDATION PASSED: all checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
