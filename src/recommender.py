"""Product recommender built with Spark MLlib's ALS.

Reads the silver event layer produced by the streaming job, converts
implicit behavior (views, clicks, purchases) into interaction strengths,
trains an ALS collaborative-filtering model, evaluates it, and writes
top-N recommendations per user to Parquet.

This is a batch training job -- in production you would retrain it on a
schedule (daily is typical) and serve the recommendations from a store.

Usage:
    python src/recommender.py --top-n 10
Run from the repo root so the `src` import resolves.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when

from config import Settings, ensure_dirs

# Implicit feedback weights: a purchase tells us much more than a view.
EVENT_WEIGHTS = {"view": 1.0, "click": 2.0, "purchase": 5.0}


def build_spark(settings: Settings) -> SparkSession:
    return (
        SparkSession.builder
        .appName(f"{settings.app_name}-recommender")
        .master(settings.spark_master)
        .config("spark.sql.shuffle.partitions", settings.shuffle_partitions)
        .getOrCreate()
    )


def load_interactions(spark: SparkSession, settings: Settings):
    """Read silver events and turn them into (user, product, rating)."""
    events = spark.read.parquet(settings.silver_path)

    # ALS needs integer ids, so hash the string ids into ints.
    # abs() keeps them non-negative, which ALS requires.
    interactions = (
        events.filter(col("event_type").isin("view", "click", "purchase"))
        .filter(col("user_id").isNotNull())
        .withColumn("user_int",
                    (abs(col("user_id").hashCode()) % 100000).cast("int"))
        .withColumn("product_int",
                    (abs(col("product_id").hashCode()) % 100000).cast("int"))
        .withColumn(
            "rating",
            when(col("event_type") == "purchase", lit(EVENT_WEIGHTS["purchase"]))
            .when(col("event_type") == "click", lit(EVENT_WEIGHTS["click"]))
            .otherwise(lit(EVENT_WEIGHTS["view"])),
        )
        # Sum strengths when a user interacted with a product repeatedly.        .groupBy("user_int", "product_int", "user_id", "product_id")
        .agg({"rating": "sum"})
        .withColumnRenamed("sum(rating)", "rating")
    )
    return interactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an ALS product recommender on silver events."
    )
    parser.add_argument("--top-n", type=int, default=10,
                        help="Recommendations per user (default: 10)")
    args = parser.parse_args()

    settings = Settings()
    ensure_dirs(settings)
    spark = build_spark(settings)
    spark.sparkContext.setLogLevel("WARN")

    interactions = load_interactions(spark, settings)
    count = interactions.count()
    print(f"Loaded {count} user-product interactions")
    if count == 0:
        print("No interactions found. Run the producer and streaming job first.")
        return

    # Time-agnostic random split is fine for a demo; a production model
    # should split by time so the test set is strictly in the future.
    train, test = interactions.randomSplit(
        [settings.train_test_split, 1 - settings.train_test_split], seed=42
    )

    als = ALS(
        userCol="user_int",
        itemCol="product_int",
        ratingCol="rating",
        rank=settings.als_rank,
        maxIter=settings.als_max_iter,
        regParam=settings.als_reg_param,
        coldStartStrategy="drop",  # avoid NaN predictions for unseen users
        nonnegative=True,
        seed=42,
    )
    print("Training ALS model...")
    model = als.fit(train)

    # Evaluate with RMSE on the held-out set.
    predictions = model.transform(test)
    rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    ).evaluate(predictions)
    print(f"Test RMSE: {rmse:.4f}")

    # Save the model and the metrics for the README / CI badges.
    model.write().overwrite().save(settings.model_dir)
    metrics_path = Path(settings.model_dir).parent / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"rmse": rmse, "interactions": count,
         "rank": settings.als_rank, "max_iter": settings.als_max_iter},
        indent=2,
    ))
    print(f"Model saved to {settings.model_dir}")

    # Top-N recommendations for every user in the training data.
    recs = model.recommendForAllUsers(args.top_n)
    # Map integer ids back to the original string ids for readability.
    id_map = interactions.select("user_int", "user_id",
                                 "product_int", "product_id").distinct()
    recs_readable = (
        recs.select("user_int", "recommendations")
        .join(id_map.select("user_int", "user_id").distinct(), "user_int")
        .select("user_id", "recommendations")
    )
    out_path = str(Path(settings.data_dir) / "gold" / "recommendations")
    recs_readable.write.mode("overwrite").parquet(out_path)
    print(f"Wrote top-{args.top_n} recommendations to {out_path}")

    print("\nSample recommendations:")
    for row in recs_readable.limit(3).collect():
        print(f"  {row['user_id']}: "
              f"{[r['product_int'] for r in row['recommendations']][:5]}")

    spark.stop()


if __name__ == "__main__":
    main()
