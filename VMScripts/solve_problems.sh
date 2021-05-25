#!/bin/bash

SOLVERS_PATH="src/solvers/"
SOLVER="GMAA-ICE"
HORIZONS=(3 4 5 6 7)
DISCOUNT=0.99
declare -a argv=$@
PROBLEMS=$argv
PROBLEMS_DIR="problems/"
LOGS_DIR="logs/"
TIMEOUT=3600
TIMEOUTFLAG="--GMAAdeadline"
#PARAMS=""
#while getopts ":t:" opt; do
#  case $opt in
#    t) timeout="$OPTARG"
#    PARAMS="$PARAMS --GMAAdeadline=$timeout"
#    ;;
#    \?) echo "Invalid option -$OPTARG" >&2
#    ;;
#  esac
#done
#
#echo $PARAMS

for problem in $PROBLEMS; do
for horizon in ${HORIZONS[@]}; do
echo "Solving problem $problem with horizon $horizon on $SOLVER"
${SOLVERS_PATH}${SOLVER} -g $DISCOUNT -h $horizon --GMAAdeadline $TIMEOUT -Q QMDP "$PROBLEMS_DIR/$problem" > "$LOGS_DIR/${problem}_h${horizon}_${SOLVER}_`date +%Y%m%d%H%M%S`.txt";
done
done
