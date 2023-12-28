if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "the number of arguments must be 2 or 3, but $# arguments were given"
    exit
fi

for f in $(find $1/camera$3*/video_??-??-??_*.mkv); do
    dir_name=$(basename $(dirname $f))
    file_name=$(basename $f)

    echo "encoding $dir_name/${file_name:0:-4}.mp4"
    ffmpeg -y -i $f -vcodec h264_nvenc -vf fps=5 $2/$dir_name/${file_name:0:-4}.mp4 2>> $2/log.txt
done
