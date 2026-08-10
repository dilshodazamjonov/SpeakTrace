# TODO

1. Extract the name from the csvs and compare for existance and save to the processed paths as parquests.

| submission_id | part | audio_paths                                                            | score | split |
|---------------|------|------------------------------------------------------------------------|-------|-------|
| SI12BD-00030  | P1   | [data/train/SI12BD-00030-P10001.flac, ...P10002.flac, ...P10003.flac]  | 4.0   | train |

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

3. Train a baseline model and model evbaluation

What i did:

1. Build a evaluation function which returns dict of metrics like: `RMSE`, `MAE`, `Bias`, `Pearson and Spearman correlations` and etc.
2. Train mean per parts and tested on Dev getting following metric results:

```
Overall RMSE: 0.6706 — primary number to beat
    Pooled part RMSE: 0.7507
    P1: 0.7870
    P3: 0.7381
    P4: 0.7267
    P5: 0.7496
```


-- Done

4. Build the audio feature extractor for `Baseline 1` and test it on a small sample

deliverable today should be something like:

| submission_id | part | duration | silence_ratio | rms_mean | mfcc_1_mean | ... | score |
|---------------|------|----------|---------------|----------|-------------|-----|-------|