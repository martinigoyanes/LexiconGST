#!/usr/bin/env bash

#SBATCH --output=jobs/%J_slurm.out
#SBATCH --error=jobs/%J_slurm.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="smaug"
#SBATCH --cpus-per-task=78
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

export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/Midgard/home/martinig/thesis-src/src
# export PYTHONPATH=$PYTHONPATH:/home/martin/Documents/Education/Master/thesis/project/thesis-src/src

DATASET_NAME=paradetox

# LEXICON_NAMES=("abusive" "hate" "ngram-refined" "toxic") 
LEXICON_NAMES=("paradetox") 
SIMILARITY_STRATEGIES=("top-1" "top-3" "top-5" "top-7" "top-9")

for LEXICON_NAME in "${LEXICON_NAMES[@]}"; do
	for SIMILARITY_STRATEGY in "${SIMILARITY_STRATEGIES[@]}"; do
		python src/preprocessing/compute_embed_sim.py --dataset_name $DATASET_NAME --lexicon_name $LEXICON_NAME --similarity_strategy $SIMILARITY_STRATEGY
	done
done