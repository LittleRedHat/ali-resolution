#!/usr/bin/bash
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
####################################

function bmp2video(){
    local input_dir=$1
    local track_id=$2
    local output_dir=$3
    local sub=$4
    if [[ ${sub} -eq 1 ]]; then
        num=1
        a=`mktemp -d`
        echo ${a}
        for file in `ls ${input_dir}/${track_id} | grep .bmp`;
        do
            cp ${input_dir}/${track_id}/${file} ${a}/`printf "%03d" ${num}`.bmp
            num=$((num+1))
        done
        ffmpeg -i ${a}/%3d.bmp  -pix_fmt yuv420p  -vsync 0 $output_dir/${track_id}_h_Sub25_Res.y4m -y
        rm -r ${a}
    else
        ffmpeg -i ${input_dir}/${track_id}/%3d.bmp  -pix_fmt yuv420p  -vsync 0 $output_dir/${track_id}_h_Res.y4m -y
    fi

}

function extract_sub(){
    input=$1
    local output_dir=$2
    local name_ext="$(basename $input)"
    local name=${name_ext%%Res\.*}
    ffmpeg -i ${input} -vf select='not(mod(n\,25))' -vsync 0  -y ${output_dir}/${name}Sub25_Res.y4m

}
input_dir=$1
output_dir=$2
top10_dir=${output_dir}/top10
another90_dir=${output_dir}/sub90
final_dir=${output_dir}/submit
usezip=${3-1}
mkdir -p ${final_dir}
full_track=(Youku_00194
            Youku_00164
            Youku_00189
            Youku_00198
            Youku_00155)

for((i=0;i<${#full_track[@]};i++));
do
    bmp2video $input_dir ${full_track[i]} $final_dir 0
done

#for track in `ls ${input_dir}`;
#do
#    contain=0
#    for((i=0;i<${#full_track[@]};i++))
#    do
#        if [[ ${full_track[i]} == $track ]]; then
#            contain=1
#            break
#        fi
#    done
#    if [[ ${contain} -eq 1 ]];then
#       echo 1
#    else
#        bmp2video ${input_dir} ${track} $final_dir 1
#    fi
#done

#for track in `ls ${input_dir}`;
#do
#
#done

#rm -rf $top10_dir
#rm -rf $another90_dir

