#!/bin/sh
#$ -cwd              # job execution in the current directory
#$ -l f_node=1       # Using f_node
#$ -l h_rt=24:00:00  # Execution time
#$ -N inference      # job execution in the current directory
          
. /etc/profile.d/modules.sh # Initialize module command
module purge

module load gcc
module load cuda openmpi/3.1.4-opa10.10
module load nccl/2.4.2
module list

python run.py --batch_size 16 --model_name AMR_Net \
    -data_dir ./dataset \
    -state_file_dir ./torch_model_MPI4/AMR_Net/rank0 \
    --padding_mode reflect --inference_mode --load_nth_state_file 0
