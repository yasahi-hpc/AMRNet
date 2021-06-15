#!/bin/sh
#$ -cwd              # job execution in the current directory
#$ -l f_node=4       # Using f_node
#$ -l h_rt=24:00:00  # Execution time
#$ -N Horovod        # job execution in the current directory
#$ -p -4
          
. /etc/profile.d/modules.sh # Initialize module command
module purge

source ~/anaconda3/etc/profile.d/conda.sh
conda activate horovod_pytorch

module load gcc
module load cuda openmpi/3.1.4-opa10.10
module load nccl/2.4.2
module list

# Run Horovod example with 4 GPUs [AT LEAST WORK]
mpirun -x PATH -x LD_LIBRARY_PATH -x PSM2_CUDA=1 -x PSM2_GPUDIRECT=1 \
    --mca pml ob1 -npernode 4 -np 4 \
    -x UCX_MEMTYPE_CACHE=n -x HOROVOD_MPI_THREADS_DISABLE=1 \
    python run.py --batch_size 4 --n_epochs 1 --model_name AMR_Net \
    -data_dir /home/1/17IKA143/jhpcn2021/work/Deeplearning/FlowCNN/SteadyFlow/AMR_Net/dataset/datasets/steady_flow_Re20_v8