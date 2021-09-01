#! /bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g gi37
#PJM --mpi proc=8

. /etc/profile.d/modules.sh # Initialize module command

module purge
module load cuda/11.1
module load nccl/2.8.4
module load gcc/8.3.1
module load ompi/4.1.1
module list

python run.py --batch_size 32 --model_name AMR_Net \
    -data_dir ./dataset \
    -state_file_dir ./torch_model_MPI4/AMR_Net/rank0 \
    --padding_mode reflect --inference_mode --load_nth_state_file 0
