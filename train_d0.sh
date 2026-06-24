#!/bin/bash
VERSION=v0
DEVICE=cuda:0 
python ./train.py --config-name ${VERSION} version=${VERSION} device=${DEVICE}
#python ./train.py --config-name ${VERSION} version=${VERSION} device=${DEVICE} resume_last=True step=300000 lr_override=0.0001
