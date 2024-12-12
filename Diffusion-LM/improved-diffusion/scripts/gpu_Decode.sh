#!/bin/bash
#SBATCH --mail-user=s2holtsh@uwaterloo.ca
#SBATCH --mail-type=begin,end,fail
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100_80g
#SBATCH --job-name="DiffusionLM"
#SBATCH --time=02:00:00
#SBATCH --output=../bashout/tests/decode/stdout-%j.log
#SBATCH --error=../bashout/tests/decode/stderr-%j.log

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

python3 ~/STAT940/Diffusion-LM/improved-diffusion/scripts/batch_decode.py ~/STAT940/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand8_transformer_lr0.0001_0.0_500_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e -1.0 ema