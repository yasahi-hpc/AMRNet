#!/bin/bash
if [ $DEVICE = "p100_tsubame3" ]; then
  qsub -g jh210049 batch_scripts/sub_tsubame3.0_p100.sh
elif [ $DEVICE = "A100" ]; then
  pjsub batch_scripts/sub_aquarius_a100.sh
else
  echo "Unregistered environment!" 
  exit
fi
