#!/bin/bash

SRC=`realpath "$1"`
DEST=`realpath "$2"`
LOGPART="$3"
CURDIR=`pwd`
DCSCN_DIR="$HOME/dcscn-super-resolution"
FFMPEG=/opt/ffmpeg-4.1-64bit-static/ffmpeg

WIDTH=720
HEIGHT=540
SCALE=2
FPS=20

DCSCN_OPTS="--scale=$SCALE --layers=8 --filters=96 --batch_num=1 --hblur_max=70 --vblur_max=70 --jpegify_min=40 --model_name=futurama-8x96-blur70-jpeg40" 

SCALED_W=$(($WIDTH * $SCALE))
SCALED_H=$(($HEIGHT * $SCALE))

cd "$DCSCN_DIR"

$FFMPEG -i "$SRC" -vf "scale=${WIDTH}:${HEIGHT}" -pix_fmt argb -c:v rawvideo -an -f image2pipe - 2> "$CURDIR/unpack-$LOGPART.log" | python sr-pipe.py $DCSCN_OPTS --pipe --img_size "${WIDTH}x${HEIGHT}" 2> "$CURDIR/upscale-$LOGPART.log" | $FFMPEG -f rawvideo -pix_fmt argb -s "${SCALED_W}x${SCALED_H}" -r $FPS -i - -pix_fmt yuv420p -c:v libx264 -crf 13 -y -preset veryslow "$DEST" 2> "$CURDIR/encode-$LOGPART.log" 
