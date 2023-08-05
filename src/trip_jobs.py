from pyspark.sql import SparkSession
from jobs.trip_analysis import trip


def main():
    # Create a SparkSession
    spark = SparkSession.builder.appName("TaxiAnalysis").getOrCreate()
    
    trip.run(spark)
    
    spark.stop()


if __name__ == "__main__":
    main()