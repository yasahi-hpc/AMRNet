#!/bin/sh
#$ -cwd              # job execution in the current directory
#$ -l f_node=4       # Using f_node
#$ -l h_rt=24:00:00  # Execution time
#$ -N Horovod        # job execution in the current directory
          
. /etc/profile.d/modules.sh # Initialize module command
module purge

module load gcc
module load cuda openmpi/3.1.4-opa10.10
module load nccl/2.4.2
module list

# For restart
if ls log_rst*.txt > /dev/null 2>&1; then
    nb_restart=$(ls log_rst*.txt | wc -l)
    echo "Submit run$nb_restart"
else
    nb_restart=0
    echo "Submit run0"
fi

# Run Horovod example with 4 GPUs [AT LEAST WORK]
if [ $nb_restart -lt 20 ]; then
    mpirun -x PATH -x LD_LIBRARY_PATH -x PSM2_CUDA=1 -x PSM2_GPUDIRECT=1 \
        --mca pml ob1 -npernode 4 -np 4 \
        -x UCX_MEMTYPE_CACHE=n -x HOROVOD_MPI_THREADS_DISABLE=1 \
        python run.py --batch_size 16 --n_epochs 1 --model_name AMR_Net \
        -data_dir ./dataset \
        --padding_mode reflect --lr 0.0001 --run_number $nb_restart
fi
