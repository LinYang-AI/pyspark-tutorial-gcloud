from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month
from pyspark.sql.window import Window

# Create a SparkSession
spark = SparkSession.builder.appName("TaxiAnalysis").getOrCreate()

def load_data():
    # Load the NYC taxi data from Parquet files in Google Cloud Storage
    data = spark.read.parquet("gs://pyspark-tutorial-linyoung/data/NYC/*")
    return data

def trip_analysis(data):
    # Average duration and distance of rides by time of day, day of week, and month of year
    window = Window.orderBy("pickup_datetime")

    trip_analysis = data.withColumn("hour", hour("pickup_datetime")) \
                       .withColumn("day_of_week", dayofweek("pickup_datetime")) \
                       .withColumn("month", month("pickup_datetime"))

    avg_duration_by_hour = trip_analysis.groupBy("hour").avg("trip_duration_minutes")
    avg_distance_by_hour = trip_analysis.groupBy("hour").avg("trip_distance")

    avg_duration_by_day = trip_analysis.groupBy("day_of_week").avg("trip_duration_minutes")
    avg_distance_by_day = trip_analysis.groupBy("day_of_week").avg("trip_distance")

    avg_duration_by_month = trip_analysis.groupBy("month").avg("trip_duration_minutes")
    avg_distance_by_month = trip_analysis.groupBy("month").avg("trip_distance")

    avg_duration_by_hour.show()
    avg_distance_by_hour.show()
    avg_duration_by_day.show()
    avg_distance_by_day.show()
    avg_duration_by_month.show()
    avg_distance_by_month.show()

def popular_locations(data):
    # Identify the top 10 pickup and dropoff locations
    top_pickup_locations = data.groupBy("PULocationID").count().orderBy(col("count").desc()).limit(10)
    top_dropoff_locations = data.groupBy("DOLocationID").count().orderBy(col("count").desc()).limit(10)

    top_pickup_locations.show()
    top_dropoff_locations.show()

def tip_analysis(data):
    # Tip percentage by trip
    data = data.withColumn("tip_percentage", col("tip_amount") / col("total_amount") * 100)

    # Tips by time: Does the time of day, week, or year affect tipping behavior?
    tip_by_hour = data.groupBy(hour("pickup_datetime")).avg("tip_percentage").orderBy("hour")
    tip_by_day_of_week = data.groupBy(dayofweek("pickup_datetime")).avg("tip_percentage").orderBy("dayofweek")
    tip_by_month = data.groupBy(month("pickup_datetime")).avg("tip_percentage").orderBy("month")

    tip_by_hour.show()
    tip_by_day_of_week.show()
    tip_by_month.show()

def fare_analysis(data):
    # Can you calculate the average fare by pickup & drop-off location?
    avg_fare_by_pickup_dropoff = data.groupBy("PULocationID", "DOLocationID").avg("fare_amount")
    avg_fare_by_pickup_dropoff.show()

    # Can you calculate the average fare by Passenger count?
    avg_fare_by_passenger_count = data.groupBy("passenger_count").avg("fare_amount")
    avg_fare_by_passenger_count.show()

    # Can you correlate the fare amount and the distance trip?
    data = data.withColumn("fare_per_distance", col("fare_amount") / col("trip_distance"))

    correlation_fare_distance = data.stat.corr("fare_per_distance", "trip_distance")
    print("Correlation between fare and trip distance:", correlation_fare_distance)

def traffic_analysis(data):
    # Calculate the average speed of a trip (average trip speed in miles per hour)
    data = data.withColumn("trip_speed", col("trip_distance") / (col("trip_duration_minutes") / 60))

    # Group the average speed by trip then hour, day, week
    avg_speed_by_hour = data.groupBy(hour("pickup_datetime")).avg("trip_speed").orderBy("hour")
    avg_speed_by_day_of_week = data.groupBy(dayofweek("pickup_datetime")).avg("trip_speed").orderBy("dayofweek")
    avg_speed_by_month = data.groupBy(month("pickup_datetime")).avg("trip_speed").orderBy("month")

    avg_speed_by_hour.show()
    avg_speed_by_day_of_week.show()
    avg_speed_by_month.show()

def demand_prediction(data):
    # Feature engineering: Use the date and time of the pickups to create features for the model
    data = data.withColumn("hour_of_day", hour("pickup_datetime"))
    data = data.withColumn("day_of_week", dayofweek("pickup_datetime"))
    data = data.withColumn("month", month("pickup_datetime"))

    # Regression model: Use linear regression to predict the number of pickups in the next hour
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    assembler = VectorAssembler(inputCols=["hour_of_day", "day_of_week", "month"], outputCol="features")
    data = assembler.transform(data)

    lr = LinearRegression(featuresCol="features", labelCol="pickups_in_next_hour")
    lr_model = lr.fit(data)

    # Make predictions for the next hour based on the features
    next_hour_data = spark.createDataFrame([(18, 2, 7)], ["hour_of_day", "day_of_week", "month"])
    next_hour_data = assembler.transform(next_hour_data)
    predictions = lr_model.transform(next_hour_data)

    predicted_pickups = predictions.select("prediction").first()[0]
    print("Predicted number of pickups in the next hour:", predicted_pickups)


if __name__ == "__main__":
    data = load_data()
    trip_analysis(data)
    popular_locations(data)
    tip_analysis(data)
    fare_analysis(data)
    traffic_analysis(data)
    demand_prediction(data)