#!/usr/bin/bash
#
# imagestogif.sh
#
# A bash script that converts a series of images produced by the visual
# multigrid Python script into an animated gif.

cd ./images
convert -verbose -delay 0 -loop 0 *.png animation.gif
