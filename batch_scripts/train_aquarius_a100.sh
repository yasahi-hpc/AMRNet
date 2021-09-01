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

# For restart
if ls log_rst*.txt > /dev/null 2>&1; then
    nb_restart=$(ls log_rst*.txt | wc -l)
    echo "Submit run$nb_restart"
else
    nb_restart=0
    echo "Submit run0"
fi

# Run Horovod example with 8 GPUs
if [ $nb_restart -lt 20 ]; then
    mpirun -np 8 -map-by ppr:8:node -mca pml ob1 python3 run.py \
        --batch_size 32 --n_epochs 1 --model_name AMR_Net \
        -data_dir ./dataset \
        --padding_mode reflect --lr 0.0001 --run_number $nb_restart
fi
