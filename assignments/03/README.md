# Assignment 3: Improving Low-Resource NMT

In this assignment we use fr-en data from the Tatoeba
corpus and investigate methods for improving low-resource NMT.

Your task is to experiment with techniques for improving
low-resource NMT systems.

## Baseline

The data used to train the baseline model was prepared using
the script `preprocess_data.sh`.
This may be useful if you choose to apply subword
segmentation or a data augmentation method.

## Train Model with BPE

Before you start you need to create a folder in "assignments/03" called "bpe", here the model checkpoints and translations data will be saved in automatically.

How to preprocess your data with bpe files:

NOTE: using the bash preprocessing script should automatically create a bpe_new folder at the correct place. but if not you can manually crate it here: "data/en-fr/prepared/" and call it "bpe_new".
This will save the preprocessed files and functions as a place for extracting the files for training later on:

1. Preprocess the data with bpe and transform the file into a compatible format for training:
```
bash assignments/03/preprocess_data_new_bpe.sh
```
2. Use the preprocessed files for training and set some flags for hyperparameter tuning (set path to log file if wanted):
```
python3 train.py --data data/en-fr/prepared/bpe_new --source-lang fr --target-lang en --save-dir assignments/03/bpe/checkpoints --log-file assignments/03/bpe/exp.log --decoder-use-lexical-model True --lr 0.0005
```
3. Translate the test file using the best model checkpoint: (or last if you prefer this)
```
python3 translate.py --data data/en-fr/prepared/bpe_new --dicts data/en-fr/prepared/bpe_new --checkpoint-path assignments/03/bpe/checkpoints/checkpoint_best.pt --output assignments/03/bpe/bpe_translation.txt
```
4. detokenize the transaltion file and calculate the BLEU score using "sacrebleu":
```
bash scripts/postprocess.sh assignments/03/bpe/bpe_translation.txt assignments/03/bpe/bpe_translation.p.txt en
```