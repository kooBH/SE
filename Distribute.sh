#!/bin/zsh
#
#
TASK=mpSEv2
VERSION=v19
python src/distribute.py -c config/${TASK}/${VERSION}.yaml --default config/${TASK}/default.yaml  -v ${VERSION} --chkpt /home/nas/user/kbh/${TASK}/chkpt/${VERSION}/bestmodel.pt -t ${TASK}

#cp chkpt/${TASK}_${VERSION}.onnx ../mpANC/chkpt/
#cd ../mpANC/bin
#./RTF ${VERSION} 1 2
