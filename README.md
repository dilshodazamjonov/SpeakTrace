# TODO

1. Extract the name from the csvs and compare for existance and save to the processed paths as parquests.

| submission_id | part | audio_paths | score | split |
|---------------|------|-------------|-------|-------|
| SI12BD-00030 | P1 | [data/train/SI12BD-00030-P10001.flac, ...P10002.flac, ...P10003.flac] | 4.0 | train |

What i did: 

1. Got the tsv files from directory of `ml/data/raw/metadata/` and added headers as `submission_id` and the `score`
2. Extracted the stages(train, dev), parts and the overall scores as a new column not the raw
3. Got audios from dir of `ml/data/raw/` only the files with the extensions of `.flac`
4. Extracted from file names the `submission_id` and `part`.
5. Merged finally tsv with the audios_df on `submission_id and part`

-- DONE

2. Perform basic EDA on the Parquet manifest.

What I did:

1. Did analysis of how scores distribution differ from Train to Dev
2. Plots on score distributions 
3. does speaking time impact score positively - Gotten Answer - YES. most of the answers whcih were answered they took 5 while being at 100s but that works only in part 1 and 3


-- DONE

3. 