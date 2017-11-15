#!/bin/bash

#SBATCH --job-name=newglove
#SBATCH --partition=p100
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH --output=tfout.out
#SBATCH --error=tferr.err
#SBATCH --time=72:00:00

source /usr/share/Modules/init/bash
unset MODULEPATH
module use /lustre/usr/modulefiles/pi
module purge
module load gcc/4.4 cuda/8.0 cudnn/5.1 anaconda/2 phdf5 hdf5 opencv/2.4 protobuf gflags


source activate theano
cd /lustre/home/eefxz/liwei/codes/ai/self-critical.pytorch_2
python train.py

