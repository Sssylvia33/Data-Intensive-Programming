name := "spark_app"
 
version := "1.0"
 
scalaVersion := "2.11.8"
 
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"
libraryDependencies += "com.databricks" %% "spark-xml" % "0.4.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "com.opencsv" % "opencsv" % "4.0"
