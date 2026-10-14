"""Spark Structured Streaming job for partner product events.

Reads the raw event stream from Kafka, then:
  1. Parses the JSON payload with an explicit schema (never inferSchema
     in streaming -- a bad record shouldn't be able to shift the schema).
  2. Drops malformed rows into a quarantine sink for later inspection.
  3. Deduplicates on event_id with a watermark, so late or replayed
     partner events don't double-count.
  4. Enriches events with a static product catalog (category margins).
  5. Writes bronze (raw) and silver (clean, enriched) Parquet outputs.

Run it with:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\
        src/streaming_job.py
Run from the repo root so the `src` import below resolves.
"""

import sys
from pathlib import Path

# Make `import config` work when launched via spark-submit from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, when
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from config import Settings, ensure_dirs

# Explicit schema for the Kafka JSON payload. Keeping this in code means
# schema changes are reviewed, not silently absorbed at runtime.
EVENT_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("event_type", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("partner_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("inventory", IntegerType(), True),
    StructField("order_id", StringType(), True),
    StructField("event_time", StringType(), True),
])

def build_spark(settings: Settings) -> SparkSession:
    """Create the Spark session with the settings from config."""
    return (
        SparkSession.builder
        .appName(settings.app_name)
        .master(settings.spark_master)
        .config("spark.sql.shuffle.partitions", settings.shuffle_partitions)
        .getOrCreate()
    )


def main() -> None:
    settings = Settings()
    ensure_dirs(settings)
    spark = build_spark(settings)
    spark.sparkContext.setLogLevel("WARN")

    # 1. Read the raw Kafka stream.
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.events_topic)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", settings.max_offsets_per_trigger)
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_str", "timestamp AS kafka_ts")
        .select(from_json(col("json_str"), EVENT_SCHEMA).alias("e"), col("kafka_ts"))
        .select("e.*", "kafka_ts")
        .withColumn("event_ts", to_timestamp(col("event_time")))
    )

    # 2. Quarantine malformed rows instead of silently dropping them.
    # Anything missing an event_id or timestamp is suspicious.
    bad_rows = parsed.filter(col("event_id").isNull() | col("event_ts").isNull())
    bad_query = (
        bad_rows.writeStream
        .format("parquet")
        .option("path", str(settings.data_dir / "quarantine" / "partner_events"))
        .option("checkpointLocation",
                str(settings.checkpoint_dir / "quarantine"))
        .outputMode("append")
        .trigger(processingTime=settings.trigger_interval)
        .start()
    )

    # 3 + 4. Clean, dedup, and enrich the good rows.
    clean = (
        parsed.filter(col("event_id").isNotNull() & col("event_ts").isNotNull())
        .withWatermark("event_ts", settings.watermark_delay)
        .dropDuplicates(["event_id"])
    )

    # Enrich with margin via a simple mapping expression. A broadcast join
    # against a real dimension table would replace this at larger scale.
    margin_expr = when(col("category") == "electronics", 0.22) \
        .when(col("category") == "home", 0.35) \
        .when(col("category") == "sports", 0.30) \
        .when(col("category") == "books", 0.45) \
        .otherwise(0.40)
    enriched = clean.withColumn("margin", margin_expr) \
        .withColumn("revenue", col("price") * col("quantity").cast("double"))

    # 5a. Bronze: raw parsed events, kept as an audit trail.
    bronze_query = (
        parsed.writeStream
        .format("parquet")
        .option("path", settings.bronze_path)
        .option("checkpointLocation", str(settings.checkpoint_dir / "bronze"))
        .outputMode("append")
        .trigger(processingTime=settings.trigger_interval)
        .start()
    )

    # 5b. Silver: deduplicated, enriched events for analytics and ML.
    silver_query = (
        enriched.writeStream
        .format("parquet")
        .option("path", settings.silver_path)
        .option("checkpointLocation", str(settings.checkpoint_dir / "silver"))
        .outputMode("append")
        .trigger(processingTime=settings.trigger_interval)
        .start()
    )

    # 5c. Console sink for quick debugging while developing.
    console_query = (
        enriched.writeStream
        .format("console")
        .option("truncate", "false")
        .outputMode("append")
        .trigger(processingTime="60 seconds")
        .start()
    )

    print("Streaming started: bronze + silver + quarantine + console. "
          "Press Ctrl+C to stop.")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
