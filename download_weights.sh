#!/bin/bash

SAMPLING=$1
SDBRANCH=$2
PRINTHELP="0"

if [[ $SAMPLING != "R" && $SAMPLING != "M" ]]; then
  echo "Available sampling methods are 'R' (random) and 'M' (mirror)"
  PRINTHELP="1"
fi

if [[ $SDBRANCH != "CLA" && $SDBRANCH != "REG" ]]; then
  echo "Available systole-dyastole branch types are 'CLA' (classification) and 'REG' (regression)"
  PRINTHELP="1"
fi

if [[ $PRINTHELP == "1" ]]
then
    echo "Command should be './download_weights.sh [R,M] [CLA,REG]'"
else
    if [[ $SAMPLING == "R" && $SDBRANCH == "CLA" ]]; then
        FILE="./output/UVT_R_CLA"
        LINK="https://imperialcollegelondon.box.com/shared/static/k04ye6l67mcef6jsna934qa2q6aw2sn2"
    elif [[ $SAMPLING == "M" && $SDBRANCH == "REG" ]]; then
        FILE="./output/UVT_M_REG"
        LINK="https://imperialcollegelondon.box.com/shared/static/1qw28blyyda5q7lvnfxk86ycms2qaweh"
    elif [[ $SAMPLING == "R" && $SDBRANCH == "REG" ]]; then
        FILE="./output/UVT_R_REG"
        LINK="https://imperialcollegelondon.box.com/shared/static/qbxklno9dch1peowcfyh8e31ix65eqcy"
    elif [[ $SAMPLING == "M" && $SDBRANCH == "CLA" ]]; then
        FILE="./output/UVT_M_CLA"
        LINK="https://imperialcollegelondon.box.com/shared/static/qywn066gwklxdanas97rz03by984xczh"
    else
        echo "Command should be './download_weights.sh [R,M] [CLA,REG]'"
    fi
    mkdir -p $FILE >> /dev/null
    FILE="${FILE}/best.pt"
    curl -L $LINK --output $FILE
fi

exit 1
