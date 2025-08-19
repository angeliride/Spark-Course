import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder.appName("spark_course")
    .master("local[*]")
    .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")
    .getOrCreate()
)

load_dotenv()
DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = os.getenv("PGHOST")
DB_PORT = os.getenv("PGPORT")

DBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"

CONN_PROP = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver",
}


def load(table: str):
    return spark.read.jdbc(JDBC_URL, table=table, properties=CONN_PROP)


actor = load("actor").cache()
film = load("film").cache()
category = load("category").cache()
film_actor = load("film_actor").cache()
film_category = load("film_category").cache()
inventory = load("inventory").cache()
rental = load("rental").cache()
payment = load("payment").cache()
customer = load("customer").cache()
address = load("address").cache()
city = load("city").cache()

# query 1
films_per_category = (
    film_category.join(category, "category_id")
    .groupBy("name")
    .agg(F.countDistinct("film_id").alias("film_count"))
    .orderBy(F.desc("film_count"))
)

films_per_category.show(truncate=False)

# query 2
actors_by_rental_duration = (
    film_actor.join(actor, "actor_id")
    .join(film, "film_id")
    .groupBy("actor_id", "first_name", "last_name")
    .agg(F.sum("rental_duration").alias("total_rental_duration"))
    .orderBy(F.desc("total_rental_duration"))
    .limit(10)
)

actors_by_rental_duration.show()

# query 3
max_cost_category = (
    film.join(film_category, "film_id")
    .join(category, "category_id")
    .groupBy("name")
    .agg(F.sum("replacement_cost").alias("total_cost"))
    .orderBy(F.desc("total_cost"))
    .limit(1)
)

max_cost_category.show()

# query 4
films_not_in_inventory = (
    film.select("title")
    .distinct()
    .subtract(film.join(inventory, "film_id").select("title"))
)

films_not_in_inventory.show(n=50, truncate=False)


# query 5
children_frequency_actors = (
    actor.join(film_actor, "actor_id")
    .join(film, "film_id")
    .join(film_category, "film_id")
    .join(category, "category_id")
    .filter(F.col("name") == "Children")
    .groupBy("actor_id", "first_name", "last_name")
    .agg(F.count("film_id").alias("film_count"))
    .orderBy(F.desc("film_count"))
    .limit(3)
)

children_frequency_actors.show()

# query 6
result = (
    city.join(address, city.city_id == address.city_id)
    .join(customer, address.address_id == customer.address_id)
    .groupBy(city.city)
    .agg(
        F.sum(F.when(customer.active == 1, 1).otherwise(0)).alias("active_clients"),
        F.sum(F.when(customer.active == 0, 1).otherwise(0)).alias("inactive_clients"),
    )
    .orderBy(F.desc("inactive_clients"))
)

result.show(n=500, truncate=False)

# query 7
rental_data = (
    rental.join(customer, "customer_id")
    .join(address, "address_id")
    .join(city, "city_id")
    .join(inventory, "inventory_id")
    .join(film, "film_id")
    .join(film_category, "film_id")
    .join(category, "category_id")
    .withColumn("total_rental_hours", F.col("rental_duration") * 24)
    .groupBy("name", "city")  # c.name -> name
    .agg(F.sum("total_rental_hours").alias("total_rental_hours"))
)

grouped_a = (
    rental_data.filter(F.col("city").like("A%"))
    .groupBy("name")
    .agg(F.sum("total_rental_hours").alias("total_hours"))
    .withColumn("group_type", F.lit("A_cities"))
)

grouped_dash = (
    rental_data.filter(F.col("city").like("%-%"))
    .groupBy("name")
    .agg(F.sum("total_rental_hours").alias("total_hours"))
    .withColumn("group_type", F.lit("Dash_cities"))
)

grouped = grouped_a.unionByName(grouped_dash)
window_spec = Window.partitionBy("group_type").orderBy(F.desc("total_hours"))
ranked = grouped.withColumn("rnk", F.rank().over(window_spec))
result = ranked.filter(F.col("rnk") == 1).select(
    "group_type", F.col("name").alias("category_name"), "total_hours"
)

result.show(truncate=False)
