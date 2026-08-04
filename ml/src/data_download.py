import os
import shutil
from zipfile import ZipFile

import requests
from tqdm import tqdm

data_links = [
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.dev.01.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.dev.02.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.01.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.02.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.03.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.04.P1.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.04.P3.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.04.P4.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.04.P5.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.05.P1.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.05.P3.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.05.P4.zip",
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/data.flac.train.05.P5.zip", 
    "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/eval-data-release-20250327.zip"
]

AUDIO_EXTENSION = ".flac"


def get_zip_data(data_links: list, save_path: str = 'ml/data/raw/archives/') -> None:
    os.makedirs(save_path, exist_ok=True)

    for link in data_links:

        file_name = link.split('/')[-1]
        full_path = save_path + file_name

        print(f"Processing file named: {file_name}")

        if not os.path.exists(full_path):
            with requests.get(link, allow_redirects=True, stream=True, timeout=30) as response:

                if response.status_code  == 200:

                    total_size = int(response.headers.get("content-length", 0))
                    chunk_size = 4 * 1024 * 1024 # 4 - 8 Mib

                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=file_name,
                        leave=True
                    ) as progress_bar, open(full_path, "wb") as file:

                        for chunk in response.iter_content(chunk_size=chunk_size):

                            if chunk:

                                file.write(chunk)
                                progress_bar.update(len(chunk))

                    print(f"Download Finished, file saved at {save_path + file_name}")

                else:
                    print(f"Failed to download. Status code: {response.status_code}")

        else:
            print(f"Skipping file {file_name}")

def zip_transfer(file: str, save_dir: str, zip_path: str = 'ml/data/raw/archives/') -> None:

    # "data.flac.dev.01.zip" -> "dev01", "data.flac.train.04.P1.zip" -> "train04-P1"
    zip_name = file.removeprefix('data.flac.').removesuffix('.zip')
    subdir = zip_name.replace('.', '', 1).replace('.', '-')

    save_dir = os.path.join(save_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    with ZipFile(zip_path + file, "r") as zip_ref:

        for file_info in tqdm(zip_ref.infolist(), desc=f"Extracting {file}"):
            if file_info.filename.lower().endswith(AUDIO_EXTENSION):

                target_path = os.path.join(save_dir, os.path.basename(file_info.filename))
                with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
            else:
                print(f"Skipping file: {file_info.filename}") 

def extract_from_zip_file(zip_path: str = 'ml/data/raw/archives/', save_dir='ml/data/raw/data/'):

    all_files = [f for f in os.listdir(zip_path) if os.path.isfile(os.path.join(zip_path, f))]

    for file in all_files: 

        if 'dev' in file:
            zip_transfer(file, save_dir + 'dev/')
        elif 'train' in file:
            zip_transfer(file, save_dir + 'train/')
        elif 'eval' in file:
            zip_transfer(file, save_dir + 'eval/')
        else:
            print(f'Skipped file: {file}')


if __name__ == "__main__":
    get_zip_data(data_links)
    extract_from_zip_file()