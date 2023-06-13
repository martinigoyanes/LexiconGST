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

MODEL_NAME=BlindGST
DATAMODULE_NAME=BlindGSTDM
DATASET_NAME=jigsaw
PREPROCESS_KIND=word_embeddings
KEEP_COMMON_WORDS=false
BATCH_SIZE=4 # from gst paper
ACCUMULATE_GRAD_BATCHES=8 # makes batchsize be 32
MAX_EPOCHS=1
REFERENCE_PATH=/Midgard/home/martinig/thesis-src/data/$DATASET_NAME/test.toxic
##### ARE YOU SURE TO TRAIN ON FULL DATASET ??? ######
TRAIN_DATA_PERCENT=1.0
##### ARE YOU SURE TO TRAIN ON FULL DATASET ??? ######

LEXICON_NAMES=("abusive" "hate" "ngram-refined" "toxic" $DATASET_NAME) 
SIMILARITY_STRATEGIES=("top-1" "top-3" "top-5" "top-7" "top-9")

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

#SBATCH --output="${JOB_OUTPUT_DIR}/experiment.out"
#SBATCH --error="${JOB_OUTPUT_DIR}/experiment.out"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen|balrog|eowyn|khazadum|rivendell|shelob"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --job-name=${DATASET_NAME}-${LEXICON_NAME}-${SIMILARITY_STRATEGY}-common_words=${KEEP_COMMON_WORDS}
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB: \${SLURM_JOB_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src


export TRANSFORMERS_OFFLINE=1

echo "Batch size: ${BATCH_SIZE}"
echo "Accumulated gradient batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "Model name: ${MODEL_NAME}"
echo "Datamodule name: ${DATAMODULE_NAME}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Dataset: ${DATASET_NAME}"
echo "Preprocess kind: ${PREPROCESS_KIND}"
echo "Train data percent: ${TRAIN_DATA_PERCENT}"
echo ""
echo ""
echo "Lexicon name: ${LEXICON_NAME}"
echo "Similarity strategy: ${SIMILARITY_STRATEGY}"
echo "Keep common words: ${KEEP_COMMON_WORDS}"

echo "################################################"
echo "#####				TRAINING              ########"
echo "################################################"

if [ "$KEEP_COMMON_WORDS" = true ] ; then
	python src/train.py --keep_common_words --dataset_name $DATASET_NAME --similarity_strategy $SIMILARITY_STRATEGY --batch_size $BATCH_SIZE --datamodule_name $DATAMODULE_NAME --model_name $MODEL_NAME --default_root_dir $JOB_OUTPUT_DIR --preprocess_kind $PREPROCESS_KIND --limit_train_batches $TRAIN_DATA_PERCENT --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES --max_epochs $MAX_EPOCHS --lexicon_name $LEXICON_NAME
else
	python src/train.py --dataset_name $DATASET_NAME --similarity_strategy $SIMILARITY_STRATEGY --batch_size $BATCH_SIZE --datamodule_name $DATAMODULE_NAME --model_name $MODEL_NAME --default_root_dir $JOB_OUTPUT_DIR --preprocess_kind $PREPROCESS_KIND --limit_train_batches $TRAIN_DATA_PERCENT --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES --max_epochs $MAX_EPOCHS --lexicon_name $LEXICON_NAME
fi

echo "################################################"
echo "#####				INFERENCE              #######"
echo "################################################"

CKPT_PATH=${JOB_OUTPUT_DIR}/lightning_logs/version_\${SLURM_JOB_ID}/checkpoints
MODEL_PATH=\${CKPT_PATH}/model
TOKENIZER_PATH=\${CKPT_PATH}/tokenizer
CKPT_FILE=\$(find \${CKPT_PATH} -name "*.ckpt")
echo "Checkpoint path: \$CKPT_PATH"
echo "Checkpoint file: \$CKPT_FILE"
echo "Model path: \$MODEL_PATH"
echo "Tokenizer path: \$TOKENIZER_PATH"

if [ "$KEEP_COMMON_WORDS" = true ] ; then
	python src/inference.py --keep_common_words --checkpoint_path \$CKPT_FILE --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path \$MODEL_PATH --tokenizer_name_or_path \$TOKENIZER_PATH
else
	python src/inference.py --checkpoint_path \$CKPT_FILE --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path \$MODEL_PATH --tokenizer_name_or_path \$TOKENIZER_PATH
fi

echo "################################################"
echo "#####				EVALUATION              ######"
echo "################################################"


python evaluation/metric.py  --inputs $REFERENCE_PATH --preds "$JOB_OUTPUT_DIR/preds.txt" --batch_size 32 --out_dir $JOB_OUTPUT_DIR --task_name $DATASET_NAME

exit 0
HERE

done
done