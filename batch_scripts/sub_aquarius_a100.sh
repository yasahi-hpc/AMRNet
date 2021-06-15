#! /bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g gi37
#PJM --mpi proc=8

. /etc/profile.d/modules.sh # Initialize module command

source /work/gr21/i18048/anaconda3/etc/profile.d/conda.sh
conda activate deep

module load cuda/11.1
module load nccl/2.8.4
module load gcc/8.3.1
module load ompi/4.1.1

mpirun -np 8 -map-by ppr:8:node -mca pml ob1 python3 run.py \
    --batch_size 32 --n_epochs 1 --model_name AMR_Net
