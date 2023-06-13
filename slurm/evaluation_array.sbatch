#!/usr/bin/env bash

# Submit a job array without a physical .sbatch file using config files a HERE document.
# https://en.wikipedia.org/wiki/Here_document
# https://slurm.schedmd.com/job_array.html
# 
# Before submitting prepare a `queue` folder where each file corresponds to one config.
# Each file is called `array.<date>.<id>.yaml`. Files corresponding to succesful runs
# are deleted. If the run fails the config file is moved to an `error` folder.
#
# Variables and commands in the HERE document work like this:
# - ${RUNS_PATH}     is evaluated *now* and takes the value 
#                    from the current shell (as defined below),
#                    it's useful to pass paths and thyperparameters
# - \${SLURM_JOB_ID} is evaluated when the job starts, therefore
#                    you can access variables set by slurm
# - $(date)          is evaluated *now* and takes the value 
#                    from the current shell (as defined above)
# - \$(date)         is evaluated when the job starts, therefore
#                    you can run commands on the node

DATASET_NAME=paradetox
PREPROCESS_KIND=bert_best_head
MODEL_NAME=BlindGST
DATAMODULE_NAME=BlindGSTDM
DATA_SPLIT=dev
REFERENCE_PATH=/Midgard/home/martinig/thesis-src/data/${DATASET_NAME}/${DATA_SPLIT}.toxic

if [ "$PREPROCESS_KIND" = "word_embeddings" ] ; then

LEXICON_NAMES=("abusive" "hate" "ngram-refined" "toxic" $DATASET_NAME) 
SIMILARITY_STRATEGIES=("top-1" "top-3" "top-5" "top-7" "top-9")
KEEP_COMMON_WORDS=false


for LEXICON_NAME in "${LEXICON_NAMES[@]}"; do
for SIMILARITY_STRATEGY in "${SIMILARITY_STRATEGIES[@]}"; do

if [ "$KEEP_COMMON_WORDS" = true ] ; then
	JOB_OUTPUT_DIR=jobs/${MODEL_NAME}/${DATASET_NAME}/${PREPROCESS_KIND}/keep_common_words/${LEXICON_NAME}/${SIMILARITY_STRATEGY}
else
	JOB_OUTPUT_DIR=jobs/${MODEL_NAME}/${DATASET_NAME}/${PREPROCESS_KIND}/remove_common_words/${LEXICON_NAME}/${SIMILARITY_STRATEGY}
fi
mkdir -p $JOB_OUTPUT_DIR 

sbatch << HERE
#!/usr/bin/env bash

#SBATCH --output="${JOB_OUTPUT_DIR}/${DATA_SPLIT}_evaluation.out"
#SBATCH --error="${JOB_OUTPUT_DIR}/${DATA_SPLIT}_evaluation.out"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen|balrog|eowyn|khazadum|rivendell|shelob"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --job-name=evaluation-${DATA_SPLIT}-${DATASET_NAME}-${LEXICON_NAME}-${SIMILARITY_STRATEGY}-common_words=${KEEP_COMMON_WORDS}
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src

echo "Dataset name: ${DATASET_NAME}"
echo "Data Split: ${DATA_SPLIT}"
echo "Lexicon name: ${LEXICON_NAME}"
echo "Similarity strategy: ${SIMILARITY_STRATEGY}"
echo "Keep common words: ${KEEP_COMMON_WORDS}"
echo "Job Output directory: ${JOB_OUTPUT_DIR}"

echo "################################################"
echo "#####				INFERENCE              #######"
echo "################################################"


CKPT_FILE=$(find ${JOB_OUTPUT_DIR} -name "*.ckpt")
MODEL_PATH=$(find ${JOB_OUTPUT_DIR} -name "model")
TOKENIZER_PATH=$(find ${JOB_OUTPUT_DIR} -name "tokenizer")

echo "Checkpoint file: \${CKPT_FILE}"
echo "Model path: \${MODEL_PATH}"
echo "Tokenizer path: \${TOKENIZER_PATH}"

export TRANSFORMERS_OFFLINE=1
if [ "$KEEP_COMMON_WORDS" = true ] ; then
	python src/inference.py --keep_common_words --evaluation_data_split $DATA_SPLIT --checkpoint_path \$CKPT_FILE --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path \$MODEL_PATH --tokenizer_name_or_path \$TOKENIZER_PATH
else
	python src/inference.py --evaluation_data_split $DATA_SPLIT --checkpoint_path \$CKPT_FILE --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path \$MODEL_PATH --tokenizer_name_or_path \$TOKENIZER_PATH
fi


echo "################################################"
echo "#####				EVALUATION              ######"
echo "################################################"

python evaluation/metric.py  --evaluation_data_split $DATA_SPLIT --inputs $REFERENCE_PATH --preds "${JOB_OUTPUT_DIR}/${DATA_SPLIT}_preds.txt" --batch_size 32 --out_dir $JOB_OUTPUT_DIR --task_name $DATASET_NAME
exit 0
HERE

done
done

else

JOB_OUTPUT_DIR=jobs/${MODEL_NAME}/${DATASET_NAME}/${PREPROCESS_KIND}
mkdir -p $JOB_OUTPUT_DIR 

sbatch << HERE
#!/usr/bin/env bash

#SBATCH --output="${JOB_OUTPUT_DIR}/${DATA_SPLIT}_evaluation.out"
#SBATCH --error="${JOB_OUTPUT_DIR}/${DATA_SPLIT}_evaluation.out"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen|balrog|eowyn|khazadum|rivendell|shelob"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --job-name=evaluation-${DATA_SPLIT}-${DATASET_NAME}-${PREPROCESS_KIND}
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src

echo "Dataset name: ${DATASET_NAME}"
echo "Data Split: ${DATA_SPLIT}"
echo "Job Output directory: ${JOB_OUTPUT_DIR}"
echo "Preprocess kind: ${PREPROCESS_KIND}"

echo "################################################"
echo "#####				INFERENCE              #######"
echo "################################################"


CKPT_FILE=$(find ${JOB_OUTPUT_DIR} -name "*.ckpt")
MODEL_PATH=$(find ${JOB_OUTPUT_DIR} -name "model")
TOKENIZER_PATH=$(find ${JOB_OUTPUT_DIR} -name "tokenizer")

echo "Checkpoint file: \${CKPT_FILE}"
echo "Model path: \${MODEL_PATH}"
echo "Tokenizer path: \${TOKENIZER_PATH}"

export TRANSFORMERS_OFFLINE=1
# python src/inference.py --dataset_name $DATASET_NAME --evaluation_data_split $DATA_SPLIT --checkpoint_path \$CKPT_FILE --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path \$MODEL_PATH --tokenizer_name_or_path \$TOKENIZER_PATH


echo "################################################"
echo "#####				EVALUATION              ######"
echo "################################################"

python evaluation/metric.py  --evaluation_data_split $DATA_SPLIT --inputs $REFERENCE_PATH --preds "${JOB_OUTPUT_DIR}/${DATA_SPLIT}_preds.txt" --batch_size 32 --out_dir $JOB_OUTPUT_DIR --task_name $DATASET_NAME
exit 0
HERE



fi