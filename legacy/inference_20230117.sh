#!/bin/bash

DEVICE=cuda:1

DIR_IN=in
#DIR_IN=/home/data/kbh/AIG2022/mix_1ch_8kHz_2sec/

VERSION=FSN_v4

DIR_OUT=out_${VERSION}
#DIR_OUT=/home/nas/DB/AIG2022/${VERSION}_estim

#python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/SE_drone/chkpt/${VERSION}/bestmodel.pt -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}  --default config/default.yaml --mono

#python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/SE_drone/chkpt/${VERSION}/bestmodel.pt -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}  --default config/default_FSN.yaml --mono

#VERSION=FSN_v0
#DIR_OUT=out_${VERSION}
#python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/kbh/work/SE/best_model.tar -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}  --default config/default_FSN.yaml --mono

#VERSION=FSN_v1
#DIR_OUT=out_${VERSION}
#python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/kbh/work/SE/best_model.tar -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}  --default config/default_FSN.yaml --mono

VERSION=FSN_v0
DIR_OUT=out_${VERSION}
python ./src/inference.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/SE_drone/chkpt/${VERSION}/bestmodel.pt -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}  --default config/default_FSN.yaml --mono


