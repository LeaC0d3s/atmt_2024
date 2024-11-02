#!/bin/bash

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed/
mkdir -p $data/prepared/bpe_new

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
done

# remove tmp files
rm $data/preprocessed/train.$src.p
rm $data/preprocessed/train.$tgt.p

# preprocess all files for model training
subword-nmt learn-bpe --input $data/preprocessed/train.fr --output $data/prepared/bpe_new/codes.fr --symbols 10000
subword-nmt learn-bpe --input $data/preprocessed/train.en --output $data/prepared/bpe_new/codes.en --symbols 10000

subword-nmt get-vocab --input $data/preprocessed/train.fr --output $data/prepared/bpe_new/dict.fr
subword-nmt get-vocab --input $data/preprocessed/train.en --output $data/prepared/bpe_new/dict.en

for split in train tiny_train test valid
  do
    subword-nmt apply-bpe --input $data/preprocessed/$split.$src --codes $data/prepared/bpe_new/codes.$src --output $data/prepared/bpe_new/$split.$src --vocabulary $data/prepared/bpe_new/dict.$src
    subword-nmt apply-bpe --input $data/preprocessed/$split.$tgt --codes $data/prepared/bpe_new/codes.$tgt --output $data/prepared/bpe_new/$split.$tgt --vocabulary $data/prepared/bpe_new/dict.$tgt
  done
echo "done!"

python3 preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/bpe_new --train-prefix $data/prepared/bpe_new/train --valid-prefix $data/prepared/bpe_new/valid --test-prefix $data/prepared/bpe_new/test --tiny-train-prefix $data/prepared/bpe_new/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 --vocab-src $data/prepared/bpe_new/dict.fr --vocab-trg $data/prepared/bpe_new/dict.en
