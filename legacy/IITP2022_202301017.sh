#!/bin/bash

DEVICE=cuda:1

DIR_IN=/home/data2/kbh/AIG/IITP/2021_NC_SSL_eval/
DIR_OUT=/home/nas/DB/AIG2022/2021_NC_SSL_eval_FSN/
VERSION=FSN_v0

for level in 0db m5db m10db m15db m20db;
do
python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt best_model.tar -d ${DEVICE} -i ${DIR_IN}/${level} -o ${DIR_OUT}/${level}  --default config/default_FSN.yaml
done

