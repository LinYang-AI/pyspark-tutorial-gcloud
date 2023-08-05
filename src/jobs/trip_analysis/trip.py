from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import Row
from pyspark.sql.functions import hour, dayofweek, month, col
from datetime import timedelta

# Create a SparkSession
# spark = SparkSession.builder.appName("TaxiAnalysis").getOrCreate()


def load_data(path, spark):
    """
    column names: VendorID, tpep_pickup_datetime, tpep_dropoff_datetime,
    passenger_count, trip_distance, RatecodeID, store_and_fwd_flag,
    PULocationID, DOLocationID, payment_type, fare_amount, extra, mta_tax,
    tip_amount, tolls_amount, improvement_surcharge, total_amount,
    congestion_surcharge, airport_fee
    """
    # Load the NYC taxi data from Parquet files
    data_df = spark.read.parquet(path)

    return data_df


def trip_analysis(taxi_df):
    
    taxi_df = taxi_df.withColumn("trip_duration", F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))
    taxi_df = taxi_df.withColumn("avg_distance", (F.col("trip_distance") / F.col("passenger_count")))
    taxi_df = taxi_df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    taxi_df = taxi_df.withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
    taxi_df = taxi_df.withColumn("pickup_month", F.month("tpep_pickup_datetime"))
    agg_df = taxi_df.groupBy("pickup_hour", "pickup_day_of_week", "pickup_month").agg(F.avg("trip_duration").alias("avg_duration"),F.avg("avg_distance").alias("avg_distance")).orderBy("pickup_hour", "pickup_day_of_week", "pickup_month")
    
    pickup_locations = taxi_df.groupBy("PULocationID").count().orderBy(F.desc("count"))
    top_pickup_locations = pickup_locations.limit(10)
    dropoff_locations = taxi_df.groupBy("DOLocationID").count().orderBy(F.desc("count"))
    top_dropoff_locations = dropoff_locations.limit(10)
    
    agg_df.show()
    print("Top 10 Pickup Locations:")
    top_pickup_locations.show()
    print("Top 10 Dropoff Locations:")
    top_dropoff_locations.show()
    return taxi_df


def tip_analysis(taxi_df):
    
    tip_analysis_df = taxi_df.withColumn("tip_percentage", F.col("tip_amount") / F.col("total_amount") * 100)
    tip_analysis_df = tip_analysis_df.withColumn("tip_percentage", F.when(F.col("tip_percentage") <= 100, F.col("tip_percentage")).otherwise(0))  # Handle possible outliers

    # Group by pickup and dropoff locations and calculate average tip percentage and average distance
    tip_location_analysis = tip_analysis_df.groupBy("PULocationID", "DOLocationID").agg(F.avg("tip_percentage").alias("avg_tip_percentage"), F.avg("trip_distance").alias("avg_distance")).orderBy(F.desc("avg_tip_percentage"))
    tip_time_analysis = tip_analysis_df.groupBy("pickup_hour", "pickup_day_of_week", "pickup_month").agg(F.avg("tip_percentage").alias("avg_tip_percentage"), F.sum("tip_amount").alias("total_tip_amount")).orderBy("pickup_hour", "pickup_day_of_week", "pickup_month")

    
    payment_tip_analysis = tip_analysis_df.groupBy("payment_type").agg(F.avg("tip_percentage").alias("avg_tip_percentage"), F.avg("tip_amount").alias("avg_tip_amount"), F.sum("tip_amount").alias("total_tip_amount"))
    
    print("Tip Analysis by Location:")
    tip_location_analysis.show()
    print("Tip Analysis by Time:")
    tip_time_analysis.show()
    print("Payment and Tip Analysis:")
    payment_tip_analysis.show()


def fare_analysis(taxi_df):
    
    fare_location_analysis = taxi_df.groupBy("PULocationID", "DOLocationID").agg(F.avg("fare_amount").alias("avg_fare")).orderBy(F.desc("avg_fare"))
    
    fare_passenger_analysis = taxi_df.groupBy("passenger_count").agg(F.avg("fare_amount").alias("avg_fare")).orderBy("passenger_count")
    
    fare_distance_correlation = taxi_df.select(F.corr("fare_amount", "trip_distance").alias("correlation")).collect()[0]["correlation"]
    
    print("Average Fare by Pickup & Drop Location:")
    fare_location_analysis.show()
    
    print("Average Fare by Passenger Count:")
    fare_passenger_analysis.show()
    
    print("Correlation between Fare Amount and Trip Distance:")
    print("Correlation coefficient:", fare_distance_correlation)


def trafic_analysis(taxi_df):
    
    trip_speed_df = taxi_df.withColumn("trip_speed", (F.col("trip_distance") / (F.col("trip_duration") / 3600)))
    trip_time_speed_analysis = trip_speed_df.groupBy("PULocationID", "DOLocationID", "pickup_hour", "pickup_day_of_week", "pickup_month").agg(F.avg("trip_speed").alias("avg_trip_speed")).orderBy("PULocationID", "DOLocationID", "pickup_hour", "pickup_day_of_week", "pickup_month")
    
    trip_time_speed_analysis.show()


def demand_analysis(taxi_df):
    
    demand_df = taxi_df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    demand_df = demand_df.withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
    demand_df = demand_df.withColumn("pickup_month", F.month("tpep_pickup_datetime"))

    feature_columns = ["pickup_hour", "pickup_day_of_week", "pickup_month"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    demand_df = assembler.transform(demand_df)
    
    feature_columns = ["pickup_hour", "pickup_day_of_week", "pickup_month"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="input_features")
    demand_df = assembler.transform(demand_df)
    
    # Aggregate data for regression
    regression_df = demand_df.groupBy("pickup_hour").agg(F.sum("passenger_count").alias("total_pickups"),F.first("input_features").alias("features"))
    
    lr = LinearRegression(featuresCol="features", labelCol="total_pickups")
    lr_model = lr.fit(regression_df)
    
    return lr_model


def prediction_hour(lr_model, taxi_df, spark):
    
    # Get the last record in the Parquet data
    last_record = taxi_df.orderBy(F.desc("tpep_pickup_datetime")).limit(1).collect()[0]

    # Extract relevant information from the last record
    last_pickup_datetime = last_record["tpep_pickup_datetime"]
    next_pickup_datetime = last_pickup_datetime + timedelta(hours=1)

    # Extract features for the next hour
    next_hour = next_pickup_datetime.hour
    next_day_of_week = next_pickup_datetime.weekday()
    next_month = next_pickup_datetime.month

    # Create a DataFrame for the next hour
    next_pickup_row_hour = Row(pickup_hour=next_hour, pickup_day_of_week=next_day_of_week, pickup_month=next_month)
    next_pickup_df_hour = spark.createDataFrame([next_pickup_row_hour])

    # Convert columns to feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=["pickup_hour", "pickup_day_of_week", "pickup_month"], outputCol="features")

    # Transform the features and make predictions for the next hour
    next_pickup_df_hour = assembler.transform(next_pickup_df_hour)
    predictions_hour = lr_model.transform(next_pickup_df_hour)
    predicted_pickups_hour = predictions_hour.select("prediction").collect()[0]["prediction"]

    print("Predicted Pickups for Next Hour:", predicted_pickups_hour)


def prediction_day(lr_model, taxi_df, spark):
    
    # Get the last record in the Parquet data
    last_record = taxi_df.orderBy(F.desc("tpep_pickup_datetime")).limit(1).collect()[0]

    # Extract relevant information from the last record
    last_pickup_datetime = last_record["tpep_pickup_datetime"]
    next_pickup_datetime = last_pickup_datetime + timedelta(hours=1)
    
    # Calculate the next day and month
    next_day_datetime = last_pickup_datetime + timedelta(days=1)
    next_day_hour = next_day_datetime.hour
    next_day_of_week = next_day_datetime.weekday()
    next_month = next_day_datetime.month

    # Create a DataFrame for the next day
    next_pickup_row_day = Row(pickup_hour=next_day_hour, pickup_day_of_week=next_day_of_week, pickup_month=next_month)
    next_pickup_df_day = spark.createDataFrame([next_pickup_row_day])

    # Convert columns to feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=["pickup_hour", "pickup_day_of_week", "pickup_month"], outputCol="features")

    # Convert columns to feature vector using VectorAssembler
    next_pickup_df_day = assembler.transform(next_pickup_df_day)

    # Make predictions for the next day
    predictions_day = lr_model.transform(next_pickup_df_day)
    predicted_pickups_day = predictions_day.select("prediction").collect()[0]["prediction"]

    print("Predicted Pickups for Next Day:", predicted_pickups_day)
    
    
def prediction_month(lr_model, taxi_df, spark):
        
    # Get the last record in the Parquet data
    last_record = taxi_df.orderBy(F.desc("tpep_pickup_datetime")).limit(1).collect()[0]

    # Extract relevant information from the last record
    last_pickup_datetime = last_record["tpep_pickup_datetime"]
    next_pickup_datetime = last_pickup_datetime + timedelta(hours=1)
    
    # Calculate the next month
    next_month_datetime = last_pickup_datetime + timedelta(days=30)  # Assuming 30 days in a month
    next_month_hour = next_month_datetime.hour
    next_day_of_week = next_month_datetime.weekday()
    next_month = next_month_datetime.month

    # Create a DataFrame for the next month
    next_pickup_row_month = Row(pickup_hour=next_month_hour, pickup_day_of_week=next_day_of_week, pickup_month=next_month)
    next_pickup_df_month = spark.createDataFrame([next_pickup_row_month])

    # Convert columns to feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=["pickup_hour", "pickup_day_of_week", "pickup_month"], outputCol="features")

    # Convert columns to feature vector using VectorAssembler
    next_pickup_df_month = assembler.transform(next_pickup_df_month)

    # Make predictions for the next month
    predictions_month = lr_model.transform(next_pickup_df_month)
    predicted_pickups_month = predictions_month.select("prediction").collect()[0]["prediction"]

    print("Predicted Pickups for Next Month:", predicted_pickups_month)


def run(spark):
    path = "./NYC/*.parquet"
    taxi_df = load_data(path, spark)
    taxi_df = trip_analysis(taxi_df)
    tip_analysis(taxi_df)
    fare_analysis(taxi_df)
    trafic_analysis(taxi_df)
    lr_model = demand_analysis(taxi_df)
    # prediction_hour(lr_model, taxi_df, spark)
    # prediction_day(lr_model, taxi_df, spark)
    # prediction_month(lr_model, taxi_df, spark)

# if __name__ == "__main__":
#     spark
#     path = "./NYC/*.parquet"
#     taxi_df = load_data(path)
#     taxi_df = trip_analysis(taxi_df)
#     tip_analysis(taxi_df)
#     fare_analysis(taxi_df)
#     trafic_analysis(taxi_df)
#     lr_model = demand_analysis(taxi_df)
    # prediction_hour(lr_model, taxi_df)
    # prediction_day(lr_model, taxi_df)
    # prediction_month(lr_model, taxi_df)