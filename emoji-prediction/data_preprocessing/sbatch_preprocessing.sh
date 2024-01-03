#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --time=10:00:00

python -V
cp -r /scratch/s4800036/emoji-prediction/emoji-prediction/ $TMPDIR/emoji-prediction/

cd $TMPDIR/emoji-prediction/data_preprocessing
python preprocessingV2.py

ls $TMPDIR/emoji-prediction/data/

rm /scratch/s4800036/emoji-prediction/emoji-prediction/data/words_around_emoji_index.pkl
cp $TMPDIR/emoji-prediction/data/words_around_emoji_index.pkl /scratch/s4800036/emoji-prediction/emoji-prediction/data/words_around_emoji_index.pkl
rm /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_around_emoji_concatenation_of_embeddings.pkl
cp $TMPDIR/emoji-prediction/data/word_around_emoji_concatenation_of_embeddings.pkl /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_around_emoji_concatenation_of_embeddings.pkl
rm /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_before_emoji_index.pkl
cp $TMPDIR/emoji-prediction/data/word_before_emoji_index.pkl /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_before_emoji_index.pkl
rm /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_around_emoji_sum_of_embeddings.pkl
cp $TMPDIR/emoji-prediction/data/word_around_emoji_sum_of_embeddings.pkl /scratch/s4800036/emoji-prediction/emoji-prediction/data/word_around_emoji_sum_of_embeddings.pkl
rm /scratch/s4800036/emoji-prediction/emoji-prediction/data/sequential_data.pkl
cp $TMPDIR/emoji-prediction/data/sequential_data.pkl /scratch/s4800036/emoji-prediction/emoji-prediction/data/sequential_data.pkl
