import org.apache.spark.mllib.clustering._

val rawData = sc.textFile("hdfs:///user/ds/ch05/kddcup.data")

val attackTypes = rawData.map(_.split(',').last).countByValue().toSeq
attackTypes.length
attackTypes.sortBy(_._2).reverse.foreach(println)

// Parser

import org.apache.spark.mllib.linalg._

val labelsAndData = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  buffer.remove(1, 3)
  val label = buffer.remove(buffer.length-1)
  val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
  (label,vector)
}

val data = labelsAndData.values.cache()

labelsAndData.take(2).foreach( println )
data.take(2).foreach( println )

// Create a KMeansModel

import org.apache.spark.mllib.clustering._

val kmeans = new KMeans()

val model = kmeans.run(data)

model.clusterCenters.foreach(println)

// Cluser lable check

val clusterLabelCount = labelsAndData.map { case (label,datum) =>
  val cluster = model.predict(datum)
  (cluster,label)
}.countByValue

clusterLabelCount.toSeq.sorted.foreach {
  case ((cluster,label),count) =>
  println(f"$cluster%1s$label%18s$count%8s")
}


// Choosing k

def distance(a: Vector, b: Vector) =
  math.sqrt(a.toArray.zip(b.toArray).
  map(p => p._1 - p._2).map(d => d * d).sum)

def distToCentroid(datum: Vector, model: KMeansModel) = {
  val cluster = model.predict(datum)
  val centroid = model.clusterCenters(cluster)
  distance(centroid, datum)
}

import org.apache.spark.rdd._

def clusteringScore(data: RDD[Vector], k: Int) = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  val model = kmeans.run(data)
  data.map(datum => distToCentroid(datum, model)).mean()
}

(5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)


def clusteringScore(data: RDD[Vector], k: Int) = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  kmeans.setRuns(10)
  kmeans.setEpsilon(1.0e-6)
  val model = kmeans.run(data)
  data.map(datum => distToCentroid(datum, model)).mean()
}

(30 to 100 by 10).par.map(k => (k, clusteringScore(data, k))).toList.foreach(println)


// sampling & save HDFS

def kModel(data: RDD[Vector], k: Int) = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  kmeans.setEpsilon(1.0e-6)
  kmeans.run(data)
}

val model = kModel(data, 100)

val sample = data.map(datum =>
    model.predict(datum) + "," + datum.toArray.mkString(",")
  ).sample(false, 0.05)

sample.take(10).foreach( println )

sample.saveAsTextFile("/user/ds/ch05/sample")

// chech dataset

val raw = sc.textFile("hdfs:///user/ds/ch05/sample")

case class NetData ( data: Array[Double] )

val datas = raw.map{ line: String =>  
  val data = line.split(',').slice(1,38).map( l => l.toDouble  )
  NetData ( data )
}

(0 until 37).map(i => datas.map(md => md.data(i)).stats()).foreach(x => println(x.max,x.min,x.mean))


// Feature Normalization

val dataAsArray = data.map(_.toArray)
val numCols = dataAsArray.first().length
val n = dataAsArray.count()

val sums = dataAsArray.reduce((a,b) => a.zip(b).map(t => t._1 + t._2))
val sumSquares =
  dataAsArray.fold(new Array[Double](numCols)) ((a,b) => a.zip(b).map(t => t._1 + t._2 * t._2))

val stdevs = sumSquares.zip(sums).map { case(sumSq,sum) => math.sqrt(n*sumSq - sum*sum)/n }
val means = sums.map(_ / n)

def normalize(datum: Vector) = {
  val normalizedArray = (datum.toArray, means, stdevs).zipped.map{
    (value, mean, stdev) => if (stdev <= 0) (value - mean) else (value - mean) / stdev }
  Vectors.dense(normalizedArray)
}


val normalizedData = data.map(normalize).cache()
(60 to 120 by 10).par.map(k =>(k, clusteringScore(normalizedData, k))).toList.foreach(println)

// 3D projection

val model = kModel(normalizedData, 100)

val sample = normalizedData.map(datum =>
    model.predict(datum) + "," + datum.toArray.mkString(",")
  ).sample(false, 0.05)

sample.take(10).foreach( println )

sample.saveAsTextFile("/user/ds/ch05/sample")

normalizedData.unpersist()

// Categorical Variables

val splitData = rawData.map(_.split(','))
val protocols = splitData.map(_(1)).distinct().collect().zipWithIndex.toMap
val services = splitData.map(_(2)).distinct().collect().zipWithIndex.toMap
val tcpStates = splitData.map(_(3)).distinct().collect().zipWithIndex.toMap

val categoryData = splitData.map { l =>
  val buffer = l.toBuffer
  val protocol = buffer.remove(1)
  val service = buffer.remove(1)
  val tcpState = buffer.remove(1)
  val label = buffer.remove(buffer.length - 1)
  val vector = buffer.map(_.toDouble)

  val newProtocolFeatures = new Array[Double](protocols.size)
  newProtocolFeatures(protocols(protocol)) = 1.0
  val newServiceFeatures = new Array[Double](services.size)
  newServiceFeatures(services(service)) = 1.0
  val newTcpStateFeatures = new Array[Double](tcpStates.size)
  newTcpStateFeatures(tcpStates(tcpState)) = 1.0

  vector.insertAll(1, newTcpStateFeatures)
  vector.insertAll(1, newServiceFeatures)
  vector.insertAll(1, newProtocolFeatures)

  (label, Vectors.dense(vector.toArray))
}

val normalizedData = categoryData.map(normalize).cache()

(60 to 200 by 10).map(k =>(k, clusteringScore(normalizedData, k))).toList.foreach(println)


// Using Labels with Entropy

def entropy(counts: Iterable[Int]) = {
  val values = counts.filter(_ > 0)
  val n: Double = values.sum
  values.map { v =>
    val p = v / n
    -p * math.log(p)
  }.sum
}

def clusteringScore(normalizedLabelsAndData: RDD[(String,Vector)], k: Int) = {

  val kmeans = new KMeans()
  kmeans.setK(k)
  kmeans.setEpsilon(1.0e-6)
  val model = kmeans.run(normalizedLabelsAndData.values)

  val labelsAndClusters = normalizedLabelsAndData.mapValues(model.predict)
  val clustersAndLabels = labelsAndClusters.map(_.swap)
  val labelsInCluster = clustersAndLabels.groupByKey().values
  val labelCounts = labelsInCluster.map(_.groupBy(l => l).map(_._2.size))

  val n = normalizedLabelsAndData.count()
  labelCounts.map(m => m.sum * entropy(m)).sum / n
}


val normalizedLabelsAndData = labelsAndData.mapValues(normalize).cache()

(60 to 200 by 10).par.map(k =>(k, clusteringScore(normalizedLabelsAndData, k))).toList.
  foreach(println)

// Clustering in Action

val kmeans = new KMeans()
kmeans.setK(150)
kmeans.setEpsilon(1.0e-6)
val model = kmeans.run(normalizedLabelsAndData.values)

val clustersAndLabels = normalizedLabelsAndData.mapValues(model.predict).map(_.swap).map( ( _, 1 ) ).countByKey

clustersAndLabels.take(10).foreach(println)

val clusterLabelCount = normalizedLabelsAndData.map { case (label,datum) =>
  val cluster = model.predict(datum)
  (cluster,label)
}.countByValue

clustersAndLabels.toSeq.sorted.foreach {
  case ((cluster,label),count) => println(f"$cluster%1s$label%18s$count%8s")
}

// Threshold

val distances = normalizedData.map(datum => distToCentroid(datum, model)).cache()
val threshold = distances.top(100).last


val lineData = rawData.map { line:String =>
  val buffer = line.split(',').toBuffer
  val protocol = buffer.remove(1)
  val service = buffer.remove(1)
  val tcpState = buffer.remove(1)
  val label = buffer.remove(buffer.length - 1)
  val vector = buffer.map(_.toDouble)

  val newProtocolFeatures = new Array[Double](protocols.size)
  newProtocolFeatures(protocols(protocol)) = 1.0
  val newServiceFeatures = new Array[Double](services.size)
  newServiceFeatures(services(service)) = 1.0
  val newTcpStateFeatures = new Array[Double](tcpStates.size)
  newTcpStateFeatures(tcpStates(tcpState)) = 1.0

  vector.insertAll(1, newTcpStateFeatures)
  vector.insertAll(1, newServiceFeatures)
  vector.insertAll(1, newProtocolFeatures)

  (line, Vectors.dense(vector.toArray))
}.cache()

val anomalies = lineData.map{ case (original, datum) =>
  val normalized = normalize(datum)
  val dist = distToCentroid(normalized, model)
  (dist,original)
}.filter{ case (dist,original) => ( dist > threshold ) }

anomalies.foreach{ case (x,y) => println(y) }
anomalies.sortByKey(true).take(10).foreach{ case (x,y) => println(y) }

