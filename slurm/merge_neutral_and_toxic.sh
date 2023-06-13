# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src

export PYTHONPATH=/Midgard/home/martinig/thesis-src/src
# export PYTHONPATH=$PYTHONPATH:/home/martin/Documents/Education/Master/thesis/project/thesis-src/src


DATASET_NAME=paradetox
LEXICON_NAMES=("abusive" "hate" "ngram-refined" "toxic" $DATASET_NAME) 
SIMILARITY_STRATEGIES=("top-1" "top-3" "top-5" "top-7" "top-9")
SPLITS=("dev" "train")
# WE_PATH=/Midgard/home/martinig/thesis-src/data/$DATASET_NAME/word_embeddings/keep_common_words
WE_PATH=/Midgard/home/martinig/thesis-src/data/$DATASET_NAME/word_embeddings/remove_common_words
BERT_PATH=/Midgard/home/martinig/thesis-src/data/$DATASET_NAME/bert_best_head_removal

# Copies from the bert_best_head_removal folder all neutral data: dev.neutral. train.neutral, test.neutral to the
# ${dataset}/${lexicon}/${strategy} folder and then merges train.neutral and train.toxic into train
# and then merges dev.neutral and dev.toxic into dev
# and then generates the test.toxic.in file in that folder
# and then generates the dev.toxic.in file in that folder
for LEXICON_NAME in "${LEXICON_NAMES[@]}"; do
	for SIMILARITY_STRATEGY in "${SIMILARITY_STRATEGIES[@]}"; do
		cp $BERT_PATH/*.neutral "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/"
		for split in "${SPLITS[@]}"; do
			rm "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/${split}"
			cat "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/${split}.neutral" >> "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/${split}"
			cat "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/${split}.toxic" >> "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/${split}"
		done
		# Generate TEST toxic input for evaluation
		rm "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/test.toxic.in"
		python /Midgard/home/martinig/thesis-src/src/utils.py --filepath "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/test.toxic" --out_dir "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}" --tgt_style NEU
		# Generate DEV toxic input for evaluation
		rm "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/dev.toxic.in"
		python /Midgard/home/martinig/thesis-src/src/utils.py --filepath "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}/dev.toxic" --out_dir "${WE_PATH}/${LEXICON_NAME}/${SIMILARITY_STRATEGY}" --tgt_style NEU
	done
done