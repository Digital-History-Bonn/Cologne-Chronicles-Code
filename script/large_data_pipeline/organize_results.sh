DATA_PATH=$1
NAME=$2
TIME=$(date +%s)

cd $DATA_PATH
mkdir xml_results
mv images/page/* xml_results
zip -r "xml_results/${NAME}_${TIME}.zip" images/page/
rm images/*.jpg
rm images/get_images.sh
rm -rf images/page