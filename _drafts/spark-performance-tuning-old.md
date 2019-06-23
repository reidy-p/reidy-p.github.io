---
layout: post
title: Tuning Spark Executors Part 2 (old)
date: 2019-05-08
---

I have used Apache Spark at work for a couple of months and have been very impressed by its performance on large volumes of data. However, I have often found the settings that control the executors in ``spark-submit`` a bit mysterious. In particular, I have never fully understood how to choose the number of executors, number of executor cores, and executor memory to improve performance. So I decided that I would run some experiments with different ``spark-submit`` settings and document the results to try to understand the effects. I will borrow from the examples in this [interesting talk](https://www.youtube.com/watch?v=vfiJQ7wg81Y) by Mark Grover and Ted Malaska at Spark Summit 2016.


Cluster Setup
---
In the Spark Summit 2016 talk linked to above the following cluster is used as a basis for all the examples:

![jpg](/static/spark-cluster-example.jpg)

I decided to use the [AWS Elastic Map Reduce](https://aws.amazon.com/emr/) service to launch clusters to run the experiments on. I will use the data on New York City taxi trips that is publicly available for free on [Amazon S3](https://registry.opendata.aws/nyc-tlc-trip-records-pds/).

In the Spark Summit talk four options for the spark settings are discussed.

### Option 1: Most Granular
The first suggestion is to have a large number of executors each with high memory. Specifically, give each executor only 1 core which means that there will be 16 executors on each node and each of therefore each of these executors will get 64GB / 16 = 4GB of memory each. Each node has 16 executors (with 1 core each) so the whole cluster has 16 executors x 6 nodes = 96 executors in total.

```terminal
$ spark-submit \
--master yarn \
--class SparkPerformanceTuning.SparkPerformanceTuning \
--num-executors 96 \
--executor-cores 1 \
--executor-memory 4g \
--jars ./jars/spark-performance-tuning.jar \
./jars/spark-performance-tuning.jar
```
The problem with this setup is that it fails to make use of the benefits of running multiple tasks in the same executor. 

### Option 2: Least Granular
The second option goes to the opposite extreme and tries to get full benefit from running multiple tasks in the same executor. In this case we allocate 1 executor per node and give this executor all of the 64GB of memory in each case. This single executor also uses all 16 cores available on each node. 

```terminal
$ spark-submit \
--master yarn \
--class SparkPerformanceTuning.SparkPerformanceTuning \
--num-executors 6 \
--executor-cores 16 \
--executor-memory 64g \
--jars ./jars/spark-performance-tuning.jar \
./jars/spark-performance-tuning.jar
```

One major issue with this setup, however, is that we need to leave some of the cluster resources for OS/Hadoop daemons.

### Option 3: Least Granular with with Overhead Allowance
We can modify the settings from Option 2 above and allocate 1GB of memory and 1 core from each node for OS/Hadoop overhead. This leaves us with the following ``spark-submit`` settings: 

```terminal
$ spark-submit \
--master yarn \
--class SparkPerformanceTuning.SparkPerformanceTuning \
--num-executors 6 \
--executor-cores 15 \
--executor-memory 63g \
--jars ./jars/spark-performance-tuning.jar \
./jars/spark-performance-tuning.jar
```

### Option 4: Further Optimisations
Mark and Ted suggest some further changes to the settings to achieve optsmal performance. First, they note that the ``--executor-memory`` setting controls the heap size but we need to reserve some more memory for off-heap memory in yarn. Second, YARN requires a core. Finally, they suggest that 15 cores per executor can lead to bad HDFS I/O throughput and advise to keep the number of cores per executor to 5 or fewer. So if we leave 1 core per enode for Hadoop/Yarn daemon cores we have 6 x 15 = 90 cores in total in our cluster. We only want a maximum of 5 cores per executor which gives us 90 cores / 5 cores per executor = 18 executors. We have 6 nodes in total so this means we will have 3 executors on each node with 63GB / 3 exeuctors = 21GB of memory each after leaving 1GB of memory on each node for overhead. We then make some final adjustments to allow for off-heap memory by witholding approximately 7% of the memory on each node leaving us with 21GB - (21 x 0.07) = 19GB for each node. We also keep 1 executor for the YARN Application Manager. This leaves us with the following settings:

```terminal
$ spark-submit \
--master yarn \
--class SparkPerformanceTuning.SparkPerformanceTuning \
--num-executors 17 \
--executor-cores 5 \
--executor-memory 19g \
--jars ./jars/spark-performance-tuning.jar \
./jars/spark-performance-tuning.jar
```

Scala Code
---
The Scala code that I used to test the performance is shown below. The code simply reads in the csv files for each month for the yellow taxi data, combines them into one dataframe, and performs some more computationally intense steps such as sorting and group by. I compiled this code locally and uploaded the outputted ``jar`` file to the cluster.

```scala
package SparkPerformanceTuning

import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.{FileStatus, Path}

object SparkPerformanceTuning {

  case class TaxiData(
                     VendorID: Int,
                     tpep_pickup_datetime: java.sql.Timestamp,
                     tpep_dropoff_datetime: java.sql.Timestamp,
                     passenger_count: Int,
                     trip_distance: Double,
                     PULocationID: Int,
                     DOLocationID: Int,
                     RatecodeID: Int,
                     store_and_fwd_flag: String,
                     payment_type: Int,
                     fare_amount: Double,
                     extra: Double,
                     mta_tax: Double,
                     improvement_surcharge: Double,
                     tip_amount: Double,
                     tolls_amount: Double,
                     total_amount: Double
                     )

  def main(args: Array[String]) = {

    val spark = SparkSession.builder.appName("SparkPerformanceTuning").getOrCreate()

    import spark.implicits._

    val conf = spark.sparkContext.hadoopConfiguration

    val nycTaxiPath = "/nyc-taxi-data/"
    val nycTaxiDataDir = new Path(nycTaxiPath)
    val nycTaxiFiles: Array[FileStatus] = nycTaxiDataDir.getFileSystem(conf).listStatus(nycTaxiDataDir)
    val nycTaxiFileNames: Seq[String] = nycTaxiFiles.map(file => nycTaxiPath ++ "/" ++ file.getPath.getName)

    val firstDF = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(nycTaxiFileNames.head)
    val otherDFs = nycTaxiFileNames.tail.map(csv => spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(csv))
    val fullDS = otherDFs.foldLeft(firstDF)((firstDF, nextDF) => firstDF.union(nextDF)).as[TaxiData]

    fullDS.printSchema
    fullDS.orderBy("total_amount").show
    println(fullDS.count)
    fullDS.groupByKey(_.PULocationID).count().show

 }
}
```

Dynamic Allocation
---
Dynamic Allocation is an optional Spark feature that was introduced to allow for the number of executors to be increased or decreased based on the current workload. This means that Spark can give resources back to the cluster when it doesn't need them and request them again as necessary. However, this dynamic allocation setting only affects the _number_ of executors and has no impact on the number of cores or memory for each executor.


```terminal
$ spark-submit \
--master yarn \
--class SparkPerformanceTuning.SparkPerformanceTuning \
--jars ./jars/spark-performance-tuning.jar \
./jars/spark-performance-tuning.jar
```
