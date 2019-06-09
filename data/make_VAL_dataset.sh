#!/usr/bin/env bash
# -*- coding: utf-8 -*-
function sample_frame(){
    local input_track=$1
    local output_file=$2
    local sample_stride=${3-5}
    local start_id=${4-1}
    local end_id=${5-100}
    for track in ${input_track};
    do
        frame_id=$((start_id))
        while [[ ${frame_id} -le ${end_id} ]];
        do
            echo ${track}_l/`printf "%03d" ${frame_id}`.bmp ${track}_h_GT/`printf "%03d" ${frame_id}`.bmp >> ${output_file}
            frame_id=$((frame_id + sample_stride))
        done
    done
}
full_track=(round1/val/Youku_00194
            round1/val/Youku_00164
            round1/val/Youku_00189
            round1/val/Youku_00198
            round1/val/Youku_00155)
val_file=./val.txt
result_file=./val_test_sisr.txt

if [[ -f ${result_file} ]];then
    rm ${result_file}
fi

for((i=0;i<${#full_track[@]};i++));
do
    echo ${full_track[i]} ${i};
done

 for((i=0;i<${#full_track[@]};i++))
    do
        sample_frame ${full_track[i]} ${result_file} 1
    done
for track in `cat ${val_file}`;
do
    contain=0
    for((i=0;i<${#full_track[@]};i++))
    do
        if [[ ${full_track[i]} == $track ]]; then
            contain=1
            break
        fi
    done
    if [[ ${contain} -eq 1 ]];then
       echo 1
    else
        sample_frame ${track} ${result_file} 25
    fi
done
