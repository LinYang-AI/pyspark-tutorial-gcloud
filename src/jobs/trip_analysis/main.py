from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# Create a SparkSession
spark = SparkSession.builder.appName("TaxiAnalysis").getOrCreate()


def load_data(path):
    """
    column names: VendorID, tpep_pickup_datetime, tpep_dropoff_datetime,
    passenger_count, trip_distance, RatecodeID, store_and_fwd_flag,
    PULocationID, DOLocationID, payment_type, fare_amount, extra, mta_tax,
    tip_amount, tolls_amount, improvement_surcharge, total_amount,
    congestion_surcharge, airport_fee
    """
    # Load the NYC taxi data from Parquet files
    data_nyc = spark.read.parquet(path)

    return data_nyc


def trip_analysis(data):
    """_summary_
    Function Task: Average duration and distance of rides: Compare these metrics by time of day, day of week, 
    and month of year. This can reveal patterns such as longer trips during rush hours, on 
    weekends, or during holiday seasons.
    """
    # Average duration and distance of rides by time of day, day of week, and month of year
    data = data.withColumn("trip_duration_minutes",
                           F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))

    data = data.withColumn("avg_distance", (F.col(
        "trip_distance") / F.col("passenger_count")))
    data = data.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    data = data.withColumn("pickup_day_of_week",
                           F.dayofweek("tpep_pickup_datetime"))
    data = data.withColumn("pickup_month", F.month("tpep_pickup_datetime"))

    agg_df = data.groupBy("pickup_hour", "pickup_day_of_week", "pickup_month").agg(F.avg("trip_duration").alias(
        "avg_duration"), F.avg("avg_distance").alias("avg_distance")).orderBy("pickup_hour", "pickup_day_of_week", "pickup_month")

    pickup_locations = data.groupBy(
        "PULocationID").count().orderBy(F.desc("count"))
    top_pickup_locations = pickup_locations.limit(10)

    dropoff_locations = data.groupBy(
        "DOLocationID").count().orderBy(F.desc("count"))
    top_dropoff_locations = dropoff_locations.limit(10)

    return agg_df, top_pickup_locations, top_dropoff_locations


def tip_analysis(data):

    tip_analysis_df = data.withColumn("tip_percentage", F.col(
        "tip_amount") / F.col("total_amount") * 100)
    tip_analysis_df = tip_analysis_df.withColumn("tip_percentage", F.when(F.col(
        "tip_percentage") <= 100, F.col("tip_percentage")).otherwise(0))  # Handle possible outliers

    # Group by pickup and dropoff locations and calculate average tip percentage and average distance
    tip_location_analysis = tip_analysis_df.groupBy(
        "PULocationID",
        "DOLocationID"
    ).agg(
        F.avg("tip_percentage").alias("avg_tip_percentage"),
        F.avg("trip_distance").alias("avg_distance")
    ).orderBy(F.desc("avg_tip_percentage"))

    tip_time_analysis = tip_analysis_df.groupBy(
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month"
    ).agg(
        F.avg("tip_percentage").alias("avg_tip_percentage"),
        F.sum("tip_amount").alias("total_tip_amount")
    ).orderBy(
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month")

    payment_tip_analysis = tip_analysis_df.groupBy("payment_type").agg(F.avg("tip_percentage").alias("avg_tip_percentage"),
                                                                       F.avg("tip_amount").alias(
                                                                           "avg_tip_amount"),
                                                                       F.sum("tip_amount").alias("total_tip_amount"))

    return tip_location_analysis, tip_time_analysis, payment_tip_analysis


def fare_analysis(data):
    fare_location_analysis = data.groupBy("PULocationID", "DOLocationID").\
        agg(F.avg("fare_amount").alias("avg_fare")).\
        orderBy(F.desc("avg_fare"))

    fare_passenger_analysis = data.groupBy("passenger_count").\
        agg(F.avg("fare_amount").alias("avg_fare")).\
        orderBy("passenger_count")
    fare_distance_correlation = data.select(F.corr("fare_amount", "trip_distance").
                                            alias("correlation")).\
        collect()[0]["correlation"]

    return fare_location_analysis, fare_passenger_analysis, fare_distance_correlation


def traffic_analysis(data):
    trip_speed_df = data.withColumn("trip_speed", (F.col(
        "trip_distance") / (F.col("trip_duration") / 3600)))

    trip_time_speed_analysis = trip_speed_df.groupBy("PULocationID",
                                                     "DOLocationID",
                                                     "pickup_hour",
                                                     "pickup_day_of_week",
                                                     "pickup_month").\
        agg(F.avg("trip_speed").
            alias("avg_trip_speed")).\
        orderBy("PULocationID",
                "DOLocationID",
                "pickup_hour",
                "pickup_day_of_week",
                "pickup_month")
    return trip_time_speed_analysis


def demand_prediction(data):

    demand_df = data.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    demand_df = demand_df.withColumn(
        "pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
    demand_df = demand_df.withColumn(
        "pickup_month", F.month("tpep_pickup_datetime"))

    feature_columns = ["pickup_hour", "pickup_day_of_week", "pickup_month"]
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features")
    demand_df = assembler.transform(demand_df)

    feature_columns = ["pickup_hour", "pickup_day_of_week", "pickup_month"]
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="input_features")
    demand_df = assembler.transform(demand_df)

    # Aggregate data for regression
    regression_df = demand_df.groupBy("pickup_hour").agg(
        F.sum("passenger_count").alias("total_pickups"),
        F.first("input_features").alias("features"))

    lr = LinearRegression(featuresCol="features", labelCol="total_pickups")
    lr_model = lr.fit(regression_df)

    return lr_model


if __name__ == "__main__":
    path = "./NYC/*.parquet"
    data = load_data(path)
    trip = trip_analysis(data)
    tip = tip_analysis(data)
    fare = fare_analysis(data)
    traffic = traffic_analysis(data)
    demand_model = demand_prediction(data)

    trip[0].show()
    print("Top 10 Pickup Locations:")
    trip[1].show()
    print("Top 10 Dropoff Locations:")
    trip[2].show()
    print("Tip Analysis by Location:")
    tip[0].show()
    print("Tip Analysis by Time:")
    tip[1].show()
    print("Payment and Tip Analysis:")
    tip[2].show()
    print("Average Fare by Pickup & Drop Location:")
    fare[0].show()
    print("Average Fare by Passenger Count:")
    fare[1].show()
    print("Correlation between Fare Amount and Trip Distance:")
    print("Correlation coefficient:", fare[2])
    traffic.show()
