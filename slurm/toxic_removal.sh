#!/usr/bin/env bash

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src

export PYTHONPATH=/Midgard/home/martinig/thesis-src/src
# export PYTHONPATH=$PYTHONPATH:/home/martin/Documents/Education/Master/thesis/project/thesis-src/src


THRESHOLD=0.7
DATASET_NAME=jigsaw
# Set to false if you want to remove all toxic-similar words even if they are commonly used words
KEEP_COMMON_WORDS=true
LEXICON_NAMES=("abusive" "hate" "ngram-refined" "toxic" $DATASET_NAME) 
SIMILARITY_STRATEGIES=("top-1" "top-3" "top-5" "top-7" "top-9")

for LEXICON_NAME in "${LEXICON_NAMES[@]}"; do
	for SIMILARITY_STRATEGY in "${SIMILARITY_STRATEGIES[@]}"; do
		echo "##########################################"
		echo "${LEXICON_NAME}   -   ${SIMILARITY_STRATEGY}  - ${THRESHOLD}"
		echo "##########################################"
		if [ "$KEEP_COMMON_WORDS" = true ] ; then
			python src/preprocessing/toxic_removal.py --keep_common_words --threshold $THRESHOLD --dataset_name $DATASET_NAME --lexicon_name $LEXICON_NAME --similarity_strategy $SIMILARITY_STRATEGY
		else
			python src/preprocessing/toxic_removal.py --threshold $THRESHOLD --dataset_name $DATASET_NAME --lexicon_name $LEXICON_NAME --similarity_strategy $SIMILARITY_STRATEGY
		fi
	done
done