DATA_PATH=$1
NAME=$2
TIME=$(date +%s)

cd $DATA_PATH

FILE_NUM=$(ls -1 images/page/*.xml 2>/dev/null | wc -l)
IMG_FILE_NUM=$(ls -1 images/*.jpg 2>/dev/null | wc -l)
if (( FILE_NUM < 1 )); then
  echo "Warning: No result data found. Skipping result saving."
  sleep 1h
else
  echo "Found ${FILE_NUM} xml result files. and ${IMG_FILE_NUM} .jpg images"
  mkdir xml_results
  zip -rj "xml_results/${NAME}_${TIME}.zip" images/page/
fi
echo "clean up images and page directory"
rm images/*.jpg
rm images/get_images.sh
rm -rf images/page
