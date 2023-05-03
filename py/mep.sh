#!/bin/bash
#
#
SAMP=$1
FOLD="LOGFILES/sample${SAMP}-out/SLAB${SAMP}/"
FILES="${FOLD}*.log"
SUB="sample${SAMP}-out/MEP/"
OUT="GoodHnums-${SAMP}.txt"
echo "#Hid: EFB: ERB: Z1: Z2: delta:\n" > $OUT
for f in $FILES
do
	python Process-NEB.py $FOLD $f $SUB $SAMP $OUT
done
