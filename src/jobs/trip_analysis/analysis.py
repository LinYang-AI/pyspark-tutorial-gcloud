from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def initialize_spark():
    spark = SparkSession.builder \
        .appName("NYC Taxi Analysis") \
        .getOrCreate()
    return spark

def load_data(spark, data_path):
    df = spark.read.option("header", "true").csv(data_path)
    return df

def trip_analysis(df):
    avg_duration_distance_by_hour = df.groupBy(hour("tpep_pickup_datetime")).agg(
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration").alias("avg_duration")
    ).orderBy("tpep_pickup_datetime")

    avg_duration_distance_by_day = df.groupBy(dayofweek("tpep_pickup_datetime")).agg(
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration").alias("avg_duration")
    ).orderBy("dayofweek(tpep_pickup_datetime)")

    avg_duration_distance_by_month = df.groupBy(month("tpep_pickup_datetime")).agg(
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration").alias("avg_duration")
    ).orderBy("month(tpep_pickup_datetime)")

    return avg_duration_distance_by_hour, avg_duration_distance_by_day, avg_duration_distance_by_month

def tip_analysis(df):
    tip_percentage_by_location = df.groupBy("PULocationID").agg(
        (avg("tip_amount") / avg("total_amount") * 100).alias("tip_percentage")
    ).orderBy("tip_percentage", ascending=False)

    tip_percentage_by_time = df.groupBy(hour("tpep_pickup_datetime")).agg(
        avg("tip_amount").alias("avg_tip_amount")
    ).orderBy("tpep_pickup_datetime")

    tip_percentage_by_payment_type = df.groupBy("payment_type").agg(
        avg("tip_amount").alias("avg_tip_amount")
    )

    return tip_percentage_by_location, tip_percentage_by_time, tip_percentage_by_payment_type

def fare_analysis(df):
    avg_fare_by_location = df.groupBy("PULocationID", "DOLocationID").agg(
        avg("fare_amount").alias("avg_fare")
    )

    avg_fare_by_passenger_count = df.groupBy("passenger_count").agg(
        avg("fare_amount").alias("avg_fare")
    )

    fare_distance_correlation = df.stat.corr("fare_amount", "trip_distance")

    return avg_fare_by_location, avg_fare_by_passenger_count, fare_distance_correlation

def traffic_analysis(df):
    df = df.withColumn("trip_speed", col("trip_distance") / col("trip_duration"))
    avg_speed_by_trip_hour = df.groupBy("PULocationID", hour("tpep_pickup_datetime")).agg(
        avg("trip_speed").alias("avg_speed")
    ).orderBy("avg_speed")

    return avg_speed_by_trip_hour

def demand_prediction(df):
    df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))

    assembler = VectorAssembler(inputCols=["pickup_hour"], outputCol="features")
    df = assembler.transform(df)

    lr = LinearRegression(featuresCol="features", labelCol="passenger_count")
    lr_model = lr.fit(df)

    return lr_model

if __name__ == "__main__":
    spark = initialize_spark()
    data_path = "NYC/*.parquet"
    df = load_data(spark, data_path)

    avg_duration_distance_by_hour, avg_duration_distance_by_day, avg_duration_distance_by_month = trip_analysis(df)
    tip_percentage_by_location, tip_percentage_by_time, tip_percentage_by_payment_type = tip_analysis(df)
    avg_fare_by_location, avg_fare_by_passenger_count, fare_distance_correlation = fare_analysis(df)
    avg_speed_by_trip_hour = traffic_analysis(df)
    lr_model = demand_prediction(df)

    print("Trip Analysis - Average Duration and Distance by Hour:")
    avg_duration_distance_by_hour.show()

    print("Trip Analysis - Average Duration and Distance by Day:")
    avg_duration_distance_by_day.show()

    print("Trip Analysis - Average Duration and Distance by Month:")
    avg_duration_distance_by_month.show()

    print("Tip Analysis - Tip Percentage by Location:")
    tip_percentage_by_location.show()

    print("Tip Analysis - Tip Percentage by Time:")
    tip_percentage_by_time.show()

    print("Tip Analysis - Tip Percentage by Payment Type:")
    tip_percentage_by_payment_type.show()

    print("Fare Analysis - Average Fare by Location:")
    avg_fare_by_location.show()

    print("Fare Analysis - Average Fare by Passenger Count:")
    avg_fare_by_passenger_count.show()

    print("Fare Analysis - Fare-Distance Correlation:")
    print("Correlation between Fare Amount and Trip Distance:", fare_distance_correlation)

    print("Traffic Analysis - Average Speed by Trip Hour:")
    avg_speed_by_trip_hour.show()

    print("Demand Prediction - Linear Regression Model Summary:")
    lr_model_summary = lr_model.summary
    print("R-squared:", lr_model_summary.r2)
    print("Root Mean Squared Error:", lr_model_summary.rootMeanSquaredError)




# if __name__ == "__main__":
#     spark = initialize_spark()
#     data_path = "path_to_your_data_folder"
#     df = load_data(spark, data_path)

#     avg_duration_distance_by_hour, avg_duration_distance_by_day, avg_duration_distance_by_month = trip_analysis(df)
#     tip_percentage_by_location, tip_percentage_by_time, tip_percentage_by_payment_type = tip_analysis(df)
#     avg_fare_by_location, avg_fare_by_passenger_count, fare_distance_correlation = fare_analysis(df)
#     avg_speed_by_trip_hour = traffic_analysis(df)
#     lr_model = demand_prediction(df)

#     avg_duration_distance_by_hour.show()
#     avg_duration_distance_by_day.show()
#     avg_duration_distance_by_month.show()
#     tip_percentage_by_location.show()
#     tip_percentage_by_time.show()
#     tip_percentage_by_payment_type.show()
#     avg_fare_by_location.show()
#     avg_fare_by_passenger_count.show()
#     print(fare_distance_correlation)
#     avg_speed_by_trip_hour.show()
#     print(lr_model.coefficients)
#     print(lr_model.intercept)
#     print(lr_model.summary.rootMeanSquaredError)
#     print(lr_model.summary.r2)
    