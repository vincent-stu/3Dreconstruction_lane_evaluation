#!/bin/bash
for file in /mnt/ve_share2/zhangwenzhi/kitti_data/raw/2011_09_28/*
do 
if [ -d "$file" ]
then
    find $file/image_00/data -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
    find $file/image_01/data -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
    find $file/image_02/data -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
    find $file/image_03/data -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
elif [ -f "$file" ]
then
    echo "$file is file!"
fi
done