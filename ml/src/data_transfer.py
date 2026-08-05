import os

import pandas as pd
from tqdm import tqdm


def merge_files_with_audios(path: str = 'ml/data/raw/metadata/', audio_path: str = 'ml/data/raw/') -> pd.DataFrame:
    """
    Gets path and audio path, retrieves needed data and merges with relevfant audios 
    """
    files = os.listdir(path)

    part_dfs, overall_dfs = [], []

    for file in files:
        if not file.endswith('.tsv'):
            continue

        split_name = file.rsplit('-sla-', 1)[0]

        if split_name.endswith('subset'):
            continue


        full_path = os.path.join(path, file)

        df = pd.read_csv(
            full_path, 
            sep='\t', 
            header=None, 
            names=['submission_id', 'score']
        )

        part = file.rsplit('-sla-', 1)[1].removesuffix('.tsv')

        df['split'] = file.split('-')[0]

        # Matching with the overall score to the different df to avoid confusion
        if part == 'overall':
            overall_dfs.append(df.rename(columns={
                'score': 'score_overall'
            }))
        else:
            df['part'] = part 
            part_dfs.append(df)


    if not part_dfs:
        print(f"No valid .tsv files found in {path}")
        return pd.DataFrame()
    
    tsv_df = pd.concat(part_dfs, ignore_index=True)
    overall_df = pd.concat(overall_dfs, ignore_index=True)

    # Merging left because tsv data is first priority data
    tsv_df = tsv_df.merge(overall_df, on=['submission_id', 'split'], how='left')

    files = [
        os.path.normpath(os.path.join(root, f))
        for root, dirs, filenames in os.walk(audio_path)
        for f in filenames
        if f.endswith('.flac')
    ]

    audio_records = []

    for p in tqdm(files, desc='Processing audio paths'):

        file = os.path.basename(p)

        file_to_compare = '-'.join(file.split('-')[:-1])
        part_prefix = file.split('-')[-1][:2]

        # DF for merging audio with the tsvs
        audio_records.append({
            'submission_id': file_to_compare,
            'part': part_prefix,
            'audio_files': p
        })

    audio_df = pd.DataFrame(audio_records)

    # one row per (submission, part), all its recordings gathered into a sorted list
    audio_df = (
        audio_df
        .sort_values('audio_files') # To keep the ordering 
        .groupby(['submission_id', 'part'], as_index=False)
        .agg(audio_files=('audio_files', list))
    )

    merged_df = tsv_df.merge(audio_df, on=['submission_id', 'part'], how='left')

    return merged_df


if __name__ == '__main__':

    merged_df = merge_files_with_audios()

    print(merged_df['audio_files'].head())
    print(merged_df.shape)
    merged_df.to_parquet('ml/data/processed/merged.parquet')
    print('Saved')