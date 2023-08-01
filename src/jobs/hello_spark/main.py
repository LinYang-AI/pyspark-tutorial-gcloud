from utils import shared_function


def analyze(spark, format="parquet", gcs_input_path=None, gcs_output_path=None):
    shared_function()
    df = spark.read.format(format).load(gcs_input_path)
    df.printSchema()
    spark.stop()
