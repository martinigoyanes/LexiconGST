#!/usr/bin/env bash

#SBATCH --output=jobs/gst/train_%J_slurm.out
#SBATCH --error=jobs/gst/train_%J_slurm.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="balrog"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
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


JOB_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/${SLURM_JOB_ID}
mkdir -p $JOB_PATH

BATCH_SIZE=4 # from gst paper
ACCUMULATE_GRAD_BATCHES=8 # makes batchsize be 32
MODEL_NAME=BlindGST
DATAMODULE_NAME=BlindGSTDM
SIMILARITY_STRATEGY=top-3
PREPROCESS_KIND=word_embeddings
LEXICON_NAME=hate
DATASET_NAME=paradetox
MAX_EPOCHS=1

##### ARE YOU SURE TO TRAIN ON FULL DATASET ??? ######
# TRAIN_DATA_PERCENT=0.1
TRAIN_DATA_PERCENT=1.0
##### ARE YOU SURE TO TRAIN ON FULL DATASET ??? ######

export TRANSFORMERS_OFFLINE=1
python src/train.py --DATASET_NAME $DATASET_NAME --similarity_strategy $SIMILARITY_STRATEGY --batch_size $BATCH_SIZE --datamodule_name $DATAMODULE_NAME --model_name $MODEL_NAME --default_root_dir $JOB_PATH --preprocess_kind $PREPROCESS_KIND --limit_train_batches $TRAIN_DATA_PERCENT --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES --max_epochs $MAX_EPOCHS --lexicon_name $LEXICON_NAME