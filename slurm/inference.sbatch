#!/usr/bin/env bash

#SBATCH --output=jobs/gst/inference_%J_slurm.out
#SBATCH --error=jobs/gst/inference_%J_slurm.out
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

PREPROCESS_KIND=word_embeddings_removal
MODEL_NAME=BlindGST
MODEL_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/365736/lightning_logs/version_365736/checkpoints/model
TOKENIZER_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/365736/lightning_logs/version_365736/checkpoints/tokenizer
DATAMODULE_NAME=ParadetoxDM

CKPT_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/365736/lightning_logs/version_365736/checkpoints/epoch=0-step=958.ckpt

export TRANSFORMERS_OFFLINE=1
python src/inference.py --checkpoint_path $CKPT_PATH --model_name $MODEL_NAME --preprocess_kind $PREPROCESS_KIND --datamodule_name $DATAMODULE_NAME --model_name_or_path $MODEL_PATH --tokenizer_name_or_path $TOKENIZER_PATH