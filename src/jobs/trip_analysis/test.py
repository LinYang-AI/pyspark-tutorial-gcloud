from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, expr
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def create_spark_session():
    return SparkSession.builder.appName("NYC_Taxi_Analysis").getOrCreate()

def load_data(spark, data_path):
    return spark.read.parquet(data_path)

def trip_analysis(data):
    trip_data = data.select(
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        hour("tpep_pickup_datetime").alias("hour"),
        dayofweek("tpep_pickup_datetime").alias("day_of_week"),
        month("tpep_pickup_datetime").alias("month")
    )

    avg_duration_distance_by_time = trip_data.groupBy("hour").agg(
        expr("avg(trip_distance)").alias("avg_distance"),
        expr("avg(unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime))").alias("avg_duration")
    )

    avg_duration_distance_by_day = trip_data.groupBy("day_of_week").agg(
        expr("avg(trip_distance)").alias("avg_distance"),
        expr("avg(unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime))").alias("avg_duration")
    )

    avg_duration_distance_by_month = trip_data.groupBy("month").agg(
        expr("avg(trip_distance)").alias("avg_distance"),
        expr("avg(unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime))").alias("avg_duration")
    )

    return avg_duration_distance_by_time, avg_duration_distance_by_day, avg_duration_distance_by_month

def tip_analysis(data):
    tip_data = data.select(
        "tpep_pickup_datetime",
        "tip_amount",
        hour("tpep_pickup_datetime").alias("hour"),
        dayofweek("tpep_pickup_datetime").alias("day_of_week"),
        month("tpep_pickup_datetime").alias("month")
    )

    tip_percentage_by_trip = tip_data.withColumn("tip_percentage", col("tip_amount") / col("fare_amount"))

    tip_percentage_by_location = tip_percentage_by_trip.groupBy("PULocationID").agg(expr("avg(tip_percentage)").alias("avg_tip_percentage"))

    tip_percentage_by_time = tip_percentage_by_trip.groupBy("hour").agg(expr("avg(tip_percentage)").alias("avg_tip_percentage"))

    # More tip analysis queries...

    return tip_percentage_by_location, tip_percentage_by_time

def fare_analysis(data):
    fare_data = data.select(
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "fare_amount",
        "trip_distance"
    )

    avg_fare_by_location = fare_data.groupBy("PULocationID", "DOLocationID").agg(expr("avg(fare_amount)").alias("avg_fare"))

    avg_fare_by_passenger_count = fare_data.groupBy("passenger_count").agg(expr("avg(fare_amount)").alias("avg_fare"))

    fare_distance_correlation = fare_data.corr("fare_amount", "trip_distance")

    return avg_fare_by_location, avg_fare_by_passenger_count, fare_distance_correlation

def traffic_analysis(data):
    traffic_data = data.select(
        "tpep_pickup_datetime",
        "trip_distance",
        hour("tpep_pickup_datetime").alias("hour"),
        dayofweek("tpep_pickup_datetime").alias("day_of_week"),
        month("tpep_pickup_datetime").alias("month")
    )

    # Calculate trip speed
    traffic_data = traffic_data.withColumn("trip_speed", col("trip_distance") / (col("tpep_dropoff_datetime").cast("long") - col("tpep_pickup_datetime").cast("long")))

    # Create window specification
    window_spec = Window.partitionBy("hour", "day_of_week", "month")

    avg_speed_by_time = traffic_data.withColumn("avg_speed", expr("avg(trip_speed) over window_spec")).select("hour", "day_of_week", "month", "avg_speed")

    # More traffic analysis queries...

    return avg_speed_by_time

def demand_prediction(data):
    demand_data = data.select("tpep_pickup_datetime", hour("tpep_pickup_datetime").alias("hour"))

    # Feature engineering
    demand_data = demand_data.withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))
    demand_data = demand_data.withColumn("month", month("tpep_pickup_datetime"))

    # Prepare features and label
    feature_columns = ["hour", "day_of_week", "month"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    demand_data = assembler.transform(demand_data)

    # Train regression model
    lr = LinearRegression(featuresCol="features", labelCol="hour_prediction")
    lr_model = lr.fit(demand_data)

    # Predict demand for next hour
    next_hour_demand = lr_model.transform(demand_data)

    return next_hour_demand

if __name__ == "__main__":
    spark = create_spark_session()
    data_path = "./NYC/*.parquet"
    data = load_data(spark, data_path)

    avg_duration_distance_by_time, avg_duration_distance_by_day, avg_duration_distance_by_month = trip_analysis(data)
    tip_percentage_by_location, tip_percentage_by_time = tip_analysis(data)
    avg_fare_by_location, avg_fare_by_passenger_count, fare_distance_correlation = fare_analysis(data)
    avg_speed_by_time = traffic_analysis(data)
    next_hour_demand = demand_prediction(data)

    # Perform further actions with the obtained results as needed
    avg_duration_distance_by_time.show()
    avg_duration_distance_by_day.show()
    avg_duration_distance_by_month.show()
    tip_percentage_by_location.show()
    tip_percentage_by_time.show()
    avg_fare_by_location.show()

    # Stop the Spark session
    spark.stop()
