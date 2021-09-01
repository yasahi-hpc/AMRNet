#!/bin/bash

args=$#
mode="train"
if [ $args -ge 1 ]; then
    mode=$1
fi

if [ $DEVICE = "p100_tsubame3" ]; then
  if [ $mode = "inference" ]; then
      qsub -g jh200051 batch_scripts/inference_tsubame3.0_p100.sh
  else
      qsub -g jh200051 batch_scripts/train_tsubame3.0_p100.sh
  fi
elif [ $DEVICE = "A100" ]; then
  if [ $mode = "inference" ]; then
      pjsub batch_scripts/inference_aquarius_a100.sh
  else
      pjsub batch_scripts/train_aquarius_a100.sh
  fi
else
  echo "Unregistered environment!" 
  exit
fi
