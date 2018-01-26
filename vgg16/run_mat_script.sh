DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=dogs_vs_cats_vgg16_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name dogs_vs_cats.finetuning \
  --package-path dogs_vs_cats/ \
  --region us-central1 \
  --scale-tier basic-gpu \
  -- \
  --gs-download True

