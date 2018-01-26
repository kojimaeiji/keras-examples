DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=cat_dog_age_vgg16_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
#rm -rf /Users/saboten/mljob/*
rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name dogs_vs_cats.finetuning \
--package-path dogs_vs_cats/ \
-- \
--gs-download True
