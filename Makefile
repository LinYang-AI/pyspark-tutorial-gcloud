include .env


create_cluster:
	gcloud dataproc clusters create cluster-epita-tutorial-linyoung \
    --region=${REGION} \
    --subnet=default \
    --enable-component-gateway \
    --optional-components=JUPYTER \
    --zone=${ZONE} \
    --master-machine-type=n1-standard-2 \
    --master-boot-disk-size=500 \
    --num-workers=2 \
    --worker-machine-type=n1-standard-2 \
    --worker-boot-disk-size=200 \
    --image-version=2.0-debian10

## sparksubmit master-node, main.jar, dependencies.zip

submit_job: build_dist copy_dist # https://cloud.google.com/sdk/gcloud/reference/dataproc/jobs/submit/pyspark
	gcloud dataproc jobs submit pyspark \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    gs://${BUCKET_NAME}/dist/${CODE_VERSION}/main.py \
	--py-files=gs://${BUCKET_NAME}/dist/${CODE_VERSION}/pyfiles.zip \
	-- \
	--job=${JOB_NAME} \
    --job-args=gcs_input_path=gs://${BUCKET_NAME}/data/NYC \
    --job-args=gcs_output_path=gs://${BUCKET_NAME}/results


create_bucket:
	gcloud storage buckets create gs://${BUCKET_NAME}

copy_nyc_dataset:
	gsutil cp data/NYC/* gs://${BUCKET_NAME}/data/NYC

copy_dist:
	gsutil cp dist/* gs://${BUCKET_NAME}/dist/${CODE_VERSION}

build_dist: clean ## build dist without dependencies
	mkdir ./dist
	cp ./src/main.py ./dist
	cd ./src && zip -x main.py -r ../dist/pyfiles.zip .

clean: clean_pyc clean_dist

clean_dist: ## delete dist folder
	rm -rf dist/ ;

clean_pyc: ## delete pyc python files
	find . -name "*pyc" -exec rm -f {} \;