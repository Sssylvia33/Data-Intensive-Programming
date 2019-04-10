package spark_app

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.SparkSession

import com.databricks.spark.xml._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}

import org.apache.spark.ml.regression.LinearRegressionModel

import java.lang.Thread
import org.apache.spark.rdd.RDD

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans

import scala.collection.JavaConversions._
import com.opencsv.CSVReader
import au.com.bytecode.opencsv.CSVWriter
import java.io._
import scala.collection.mutable.ListBuffer
import scala.util.control.Exception.Finally


object sparkApp extends App {
  
  // Suppress the log messages
  Logger.getLogger("org").setLevel(Level.OFF)

	val spark = SparkSession.builder()
                          .appName("Assignment")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()        
  val sc = spark.sparkContext
  spark.conf.set("spark.sql.shuffle.partitions", "5")

  val accidentDataRaw: DataFrame = spark.read
                                       .option("delimiter", ";")
                                       .option("header", "true")
                                       .option("inferSchema", "true")
                                       .option("encoding", "ISO-8859-1")
                                       .csv("data/tieliikenneonnettomuudet_2015_onnettomuus.csv")
                             
  // Select X,Y and Weekday
  val accidentData: DataFrame = accidentDataRaw.select("X", "Y","Vkpv")

  // Cleaning data having invalid coordinates 
  val cleanData: DataFrame= accidentData.where(col("Y").isNotNull)
                                        .where(col("X").isNotNull)
                                        .where(col("Vkpv").isNotNull)
                       
  println("Setup defined")

  
  // Write data to CSV file
  def writeToCSV(filename: String, data: ListBuffer[Array[String]]) = {
    val outputFile = new BufferedWriter(new FileWriter(filename))
    val csvWriter = new CSVWriter(outputFile)
    csvWriter.writeAll(data.toList)
    outputFile.close()
    println("Results have been written to a CSV file")
  }
  
  // Return training data for only with (X,Y) columns 
  def getTrData() : DataFrame = {   
    // Setting pipeline for extracting feature
    val vectorAssembler = new VectorAssembler()
                              .setInputCols(Array("X", "Y"))
                              .setOutputCol("features")
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(cleanData)
    val trData = pipeLine.transform(cleanData)  
    
    return trData
  }
  
  // Convert weekday from string to number
  val dayColToInt = udf((dayOfWeek:String) => {
	  dayToInt(dayOfWeek)
  })
  
  // Define day to be a number
  def dayToInt(weekday: String): Int = {
    weekday match {
  	  case "Sunnuntai"    => 7
  	  case "Maanantai"    => 1
  	  case "Tiistai"      => 2 
  	  case "Keskiviikko"  => 3
  	  case "Torstai"      => 4
  	  case "Perjantai"    => 5
  	  case "Lauantai"     => 6
  	  }
  }

  val PI = 3.14159265
  
  // Task 1
  def task1() {
    println("Task 1")
    
    val CLUSTERS = 4
    // Fit original data to kMeans
    val kmeans = new KMeans().setK(CLUSTERS).setSeed(1L)
    val kmModel = kmeans.fit(getTrData())
    
    // Print predictions and clusters
    println("Cluster Predictions")
    kmModel.summary.predictions.show()
    println("Cluster Center coordinates")
    kmModel.clusterCenters.foreach(println)
    
    // Write to CSV file
    var stringsToWrite = new ListBuffer[Array[String]]()
    stringsToWrite += Array("X","Y")
    kmModel.clusterCenters.foreach(center =>
      stringsToWrite += center.toArray.map(num => num.toString)
    )
    writeToCSV("results/basics.csv", stringsToWrite)
  }
  
  // Return 3 dimensional data
  def getTr3dData() : DataFrame = {
	  val numData = cleanData.withColumn("weekday",dayColToInt(col("Vkpv")))
	  println("Data, when week days have been changed to numbers")
	  numData.show(7)
	  
	  val cycleData = numData.withColumn("WeekX", bround(sin(lit(2) * lit(PI) * col("weekday") / lit(7)), 5))
	                         .withColumn("WeekY", bround(cos(lit(2) * lit(PI) * col("weekday") / lit(7)) ,5))
	  
	  // Setting pipeline for extracting feature
	  val vec3Assem = new VectorAssembler()
                      	  .setInputCols(Array("X", "Y", "WeekX", "WeekY"))
                      	  .setOutputCol("features")
	  val transPipeline2 = new Pipeline().setStages(Array(vec3Assem))
	  val vec3pipeLine = transPipeline2.fit(cycleData)
	  val tr3dData = vec3pipeLine.transform(cycleData)
	  
	  return tr3dData
  }
  
  // Return a weekday from given angle
  def getWeekDayFromAngle(angle: Double): Double = {
    if (angle < 0) { 
      return ((2 * PI + angle) * 7 / (2*PI))
    }
    else { 
      return (angle * 7 / (2*PI))
    }
  }
  
  // Return (sin, cos), function overload for int and double type weekday
  def getAngleFromWeekDay(weekDay: Int): (Double, Double) = {
    val radian = 2 * PI * weekDay / 7
    return (Math.sin(radian), Math.cos(radian))
  }
  def getAngleFromWeekDay(weekDay: Double): (Double, Double) = {
    val radian = 2 * PI * weekDay / 7
    return (Math.sin(radian), Math.cos(radian))
  }
  
  // Task 2
  def task2() {
    println("Task 2")
    
    val tr3dData = getTr3dData()
    
    // Fit original data to kMeans
    val CLUSTERS = 4
    val kmeans3d = new KMeans().setK(CLUSTERS).setSeed(1L)
    val kmModel3d = kmeans3d.fit(tr3dData)
    
    // Print predictions and clusters
    println("Cluster Predictions")
    kmModel3d.summary.predictions.show()
    println("Cluster Center coordinates")
    println("In 4 dimensions")
    kmModel3d.clusterCenters.foreach(println)
    
    val cCenters = kmModel3d.clusterCenters
    val result: Array[(Double, Double, Double)] = cCenters.map(center => (center(0), center(1), getWeekDayFromAngle(Math.atan2(center(2), center(3)))))
    println("In 3 dimensions")
    result.foreach(println)
    
    // Write to CSV file
    var clusterCenter3D = new ListBuffer[Array[String]]()
    clusterCenter3D += Array("X,Y,weekday")
    result.foreach(center =>
      clusterCenter3D += Array(center._1 + "," + center._2 + "," + center._3)
    )
    writeToCSV("results/task2.csv", clusterCenter3D)
  }
  
  
  // Task 3
  def task3() {
    println("Task3")
    
        // Below is the sketch for task #3
//    import org.apache.spark.streaming.{Seconds, StreamingContext}
//
//
//
//    val ssc = new StreamingContext(sc, Seconds(5))
//
//  
//    val accidentSchemaRaw = accidentDataRaw.schema
//    
//    val accidentDataStreamingRaw: DataFrame = spark.readStream.schema(accidentSchemaRaw)
//                                                              .format("csv")
//                                                              .option("header", "true")
//                                                              .load("streamingData/*.csv")
//    // Select X and Y
//    val accidentDataStreaming: DataFrame = accidentDataStreamingRaw.selectExpr("X", "Y")
//    
//  
//    //cleaning data having invalid coordinates
//    val cleanDataStreaming: DataFrame= accidentDataStreaming.where(col("Y").isNotNull).where(col("X").isNotNull)
//    
//    val trDataStreaming = cleanDataStreaming.rdd
//    
//    val kmModelsStreaming = new StreamingKMeansModel(kmModel.clusterCenters, Array(1.0))
//    
//    val kmModelStreaming = kmeans.trainOn(trDataStreaming)
    
    println("The sketch is in the source code comments under Task #3")
  }
  
  
  // Task 4
  def task4() {
    println("Task 4")
    val START = 10
    val STOP = 500
    val STEP = 10
    
    // Task 1 data
    val trData = getTrData()
    println("Loop task 1 clusters in steps of " + STEP)
    for (k <- START.to(STOP).by(STEP)) {
      val kmeans = new KMeans().setK(k).setSeed(System.nanoTime.toInt)
      val kmModel = kmeans.fit(trData)
  	  println(k.toString + ";" + kmModel.computeCost(trData))
    }
    
    // Task 2 data
    val tr3dData = getTr3dData()
    println("Loop task 2 clusters in steps of " + STEP)
    for (k <- START.to(STOP).by(STEP)) {
      val kmeans = new KMeans().setK(k).setSeed(System.nanoTime.toInt)
      val kmModel = kmeans.fit(tr3dData)
  	  println(k.toString + ";" + kmModel.computeCost(tr3dData))
    }
  }
  
  case class Accident(id: String,
                      X: Double,
                      Y: Double,
                      wkday: Int)
  
  // Return data in Accident class formed RDD
  def linesToAccidents(lines: RDD[String]): RDD[Accident] = {
    val arrayRDD: RDD[Array[String]] = lines.map{ line => 
      val reader = new CSVReader(new StringReader(line), ';')
      reader.readNext()
    }
    
    // Read corerct columns
    val accidentRDD: RDD[Accident] = arrayRDD.flatMap{ arr => {
      try { Some(Accident(id = arr(0),
                          X = arr(55).toDouble,
                          Y = arr(56).toDouble,
                          wkday = dayToInt(arr(19))))
      }
      catch {
        case e: Exception => None
      }
      }
    }
    
    return accidentRDD
  }
  
  
  // Get a new closest cluster mean
  def findClosest(p: (Double, Double, Double, Double), means: Array[(Double, Double, Double, Double)]): Int = {
    val distancesToMean: Array[Double] = means.map(mean => calcDistance(p, mean))
    return distancesToMean.zipWithIndex.min._2
  }
  
  // Get a new mean value for a cluster (average Accident)
  def averageAcc(i: Iterable[(Double, Double, Double, Double)]): (Double, Double, Double, Double) = {
    val N = i.size()
    val sum = i.reduce((total, acc) => ((total._1 + acc._1, total._2 + acc._2, total._3 + acc._3, total._4 + acc._4)))
    return (sum._1 / N, sum._2 / N, sum._3 / N, sum._4 / N)
  }
  
  
  // Calculate the distance from x1 to x2 using the Euclidean method for three dimensional data
  def calcDistance(x1: (Double, Double, Double, Double), x2: (Double, Double, Double, Double)): Double = {
    Math.sqrt(Math.pow(x1._1 - x2._1, 2) + Math.pow(x1._2 - x2._2, 2) + Math.pow(x1._3 - x2._3, 2) + Math.pow(x1._4 - x2._4, 2))
  }
  
  
  // Calculate mean values for k clusters
  def kMeans(data: RDD[Accident], k: Int): Array[(Double, Double, Double)] = {
    // Calculate recursively the cluster means according to the KMeans algorithm
    def kMeansWorker(means: Array[(Double, Double, Double, Double)],
                     accRDD: RDD[(Double, Double, Double, Double)],
                     iter: Int = 1,
                     k: Int): Array[(Double, Double, Double, Double)] = {
      // End recursion condition
      if (iter >= 10) return means
      
      // Calculate the new clusters for the Accidents and their new means (cluster centers)
      val m : Array[(Int, (Double, Double, Double, Double))] =
                  accRDD.map(p =>(findClosest((p._1, p._2, p._3, p._4), means), p))
                        .groupByKey()
                        .mapValues(averageAcc)
                        .collect()
  
       // Pick only the averages
       val newMeans: Array[(Double, Double, Double, Double)] = m.map(p => p._2)
       
       // Stop iteration before the cluster centers converge
       if(newMeans.length < k)return means
       return kMeansWorker(newMeans, accRDD, iter + 1, k)
    }
    
    // Start the calculation and convert days to circular form (unit circle)
    // Take random points from RDD to increase the accuracy
    val initialMeans: Array[(Double, Double, Double, Double)] = 
      data.takeSample(false, k, System.nanoTime.toInt)
          .map(d => (d.X, d.Y, getAngleFromWeekDay(d.wkday)._1, getAngleFromWeekDay(d.wkday)._2))

    val circularData = data.map(d => (d.X, d.Y, getAngleFromWeekDay(d.wkday)._1, getAngleFromWeekDay(d.wkday)._2))
    val centers = kMeansWorker(initialMeans, circularData, 1, k)
    
    // Return the result of the function call kMeansWorker but converted back to three dimensions
    val means = centers.map(center => (center._1, center._2, getWeekDayFromAngle(Math.atan2(center._3,center._4))))
    return means
  }
  
  // Task 5
  def task5() {
    println("Task 5")
    
    // Read data, drop the first line (names of the columns)
    val lines: RDD[String] = sc.textFile("data/*.csv")
                               .mapPartitionsWithIndex((i, iter) => if (i == 0) iter.drop(1)
                                                                    else iter)    
    
    val raw = linesToAccidents(lines)
    
    val CLUSTERS = 4
    val ms: Array[(Double, Double, Double)] = kMeans(raw, CLUSTERS)
    
    println("\n Cluster centers for 4 clusters as in tasks 1 and 2")
    ms.foreach(println)
    
    // Write to CSV file
    var clusterCenter3D = new ListBuffer[Array[String]]()
    clusterCenter3D += Array("X,Y,weekday")
    ms.foreach(center =>
      clusterCenter3D += Array(center._1.toString + "," + center._2.toString + "," + center._3.toString)
    )
    writeToCSV("results/task5.csv", clusterCenter3D)
  }
  
  
  // Task 6
  def task6() {
    println("Task 6")
    
    // Calculate all distances of every data point to the closest cluster center and sum them
    def totalDistance(data: RDD[(Double,Double,Double,Double,Int)],means: Array[(Double, Double, Double, Double)]): Double = {    
      val distEach = data.map(p=>(p._5,calcDistance((p._1,p._2,p._3,p._4), (means(p._5)))))
      val SSE = distEach.map(p=>p._2).reduce((x1,x2)=> x1+x2)
      
      return SSE
    }
    
    val lines = sc.textFile("data/*.csv")
                  .mapPartitionsWithIndex((i, iter) => if (i == 0) iter.drop(1)
                                                       else iter)    
    
    val accRDD: RDD[Accident] = linesToAccidents(lines)
    
    val start = 10
    val stop = 500
    val step = 10

    // (k, total_distance) keep tracking the cost for every k
    var costs: Array[(Int, Double)] = Array()
    
    // Loop different k number of clusters
    for (k <- start.to(stop).by(step) ){
      // k number of cluster centers
      // (X, Y, sin, cos)
      val means: Array[(Double, Double, Double, Double)] = kMeans(accRDD, k).map(p => (p._1, p._2, getAngleFromWeekDay(p._3)._1, getAngleFromWeekDay(p._3)._2))
      
      // Add every data point to a cluster
      // (X, Y, sin, cos, cluster_index)
      val preds: RDD[(Double,Double,Double,Double,Int)]  = accRDD.map(p => (p.X, p.Y, getAngleFromWeekDay(p.wkday)._1, getAngleFromWeekDay(p.wkday)._2))
                                                                 .map(p => (p._1, p._2, p._3, p._4, findClosest(p, means)))
      // Total cost for k clusters
      val totalCost = totalDistance(preds,means)
      println("Total distance " + "k(" + k + "): " + totalCost)
      
      // Add to costs
      costs :+= (k, totalCost)
    }
    
    println("\nCosts:")
    println("(k,total_distance)")
    costs.foreach(println)
    
    // Calculate the elbow point
    
    // Create a reference line
    val m = (costs(0)._2 - costs.last._2) / (costs(0)._1 - costs.last._1) // Slope of the line
    val b = costs(0)._2 - m*costs(0)._1 // y-intercept of the line
    
    val dist2Line = costs.map(p=>(p._1,m*p._1-p._2))
    val elbowPoint = dist2Line.maxBy(_._2)
    println("\nElbow point = " + elbowPoint._1)
    
  }
  
  // Command line arguments
  if (args.length == 0) {
    println("There was no given arguments")
  } else if (args(1) == "1") {
    task1()
  } else if (args(1) == "2") {
    task2()
  } else if (args(1) == "3") {
    task3()
  } else if (args(1) == "4") {
    task4()
  } else if (args(1) == "5") {
    task5()
  } else if (args(1) == "6") {
    task6()
  }
  
  
}
