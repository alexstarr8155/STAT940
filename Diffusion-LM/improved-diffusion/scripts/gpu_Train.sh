#!/bin/bash
#SBATCH --mail-user=s2holtsh@uwaterloo.ca
#SBATCH --mail-type=begin,end,fail
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100_80g
#SBATCH --job-name="DiffusionLM"
#SBATCH --time=02:00:00
#SBATCH --output=../bashout/tests/train/stdout-%j.log
#SBATCH --error=../bashout/tests/train/stderr-%j.log


# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

python3 ~/STAT940/Diffusion-LM/improved-diffusion/scripts/run_train.py \
    --diff_steps=500 \
    --model_arch='transformer' \
    --lr=0.0001 \
    --lr_anneal_steps=50000  \
    --seed=102 \
    --noise_schedule='sqrt' \
    --in_channel=8 \
    --modality='e2e-tgt' \
    --submit='no' \
    --padding_mode='block' \
    --app='--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data ' \
    --notes='xstart_e2e'