# zip the txt tag files to save space
DATA_DIR=../../output
TAG_FILES=$DATA_DIR/*tags.txt
for TAG_FILE in $TAG_FILES;
do
    echo $TAG_FILE
    (bzip2 $TAG_FILE)&
done