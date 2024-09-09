if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "the number of arguments must be 2 or 3, but $# arguments were given"
    exit
fi

for d in $(find $1/camera$3* -maxdepth 0); do
    dir_name=$(basename $d)
    if [ ! -d $2/$dir_name/ ]; then
        mkdir --parents $2/$dir_name/
    fi

    for f in $(find $d/video_??-??-??_*.mkv); do
        file_name=$(basename $f)
        echo "encoding $(basename $d)/${file_name:0:-4}.mp4"
        ffmpeg -y -i $f -vcodec h264_nvenc -vf fps=5 $2/$dir_name/${file_name:0:-4}.mp4 2>> $2/log.txt
        echo >> $2/log.txt
    done
done
