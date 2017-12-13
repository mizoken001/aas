val rawUserArtistData = sc.textFile("hdfs:///user/ds/ch03/user_artist_data.txt",5)

rawUserArtistData.take(10).foreach(println)

rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
rawUserArtistData.map(_.split(' ')(1).toDouble).stats()

val rawArtistData = sc.textFile("hdfs:///user/ds/ch03/artist_data.txt")

rawArtistData.take(10).foreach(println)

val artistByID = rawArtistData.map { line =>
  val (id, name) = line.span(_ != '\t')
  (id.toInt, name.trim)
}
artistByID.take(10000).foreach(println)

val artistByID = rawArtistData.flatMap { line =>
    val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
}
artistByID.take(10000).foreach(println)

val rawArtistAlias = sc.textFile("hdfs:///user/ds/ch03/artist_alias.txt",5)

rawArtistAlias.take(10).foreach(println)

val artistAlias = rawArtistAlias.flatMap { line =>
  val tokens = line.split('\t')
    if (tokens(0).isEmpty) {
      None
    } else {
      Some((tokens(0).toInt, tokens(1).toInt))
    }
}.collectAsMap()

artistByID.lookup(6803336).head
artistByID.lookup(1000010).head

import org.apache.spark.mllib.recommendation._

val bArtistAlias = sc.broadcast(artistAlias)

val trainData = rawUserArtistData.map { line =>
  val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
  val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
  Rating(userID, finalArtistID, count)
}.cache()

val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

model.rank
model.userFeatures.mapValues(_.mkString(", ")).first()
model.productFeatures.mapValues(_.mkString(", ")).first()


val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).
filter { case Array(user,_,_) => user.toInt == 2093760 }

val existingProducts = rawArtistsForUser.
                            map { case Array(_,artist,_) => artist.toInt }.
                            collect().toSet

artistByID.filter { case (id, name) => existingProducts.contains(id) }.
           values.collect().foreach(println)

val recommendations = model.recommendProducts(2093760, 5)
recommendations.foreach(println)

val recommendedProductIDs = recommendations.map(_.product).toSet
artistByID.filter { case (id, name) => recommendedProductIDs.contains(id)}.
           values.collect().foreach(println)


import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import scala.collection.Map


def buildRatings(
      rawUserArtistData: RDD[String],
      bArtistAlias: Broadcast[Map[Int,Int]]) = {
         rawUserArtistData.map { line =>
         val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
         val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
         Rating(userID, finalArtistID, count)
      }
}


:load ch03/AreaUnderCurve.scala

val allData  = buildRatings(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
trainData.cache()
cvData.cache()

val allItemIDs = allData.map(_.product).distinct().collect()
val bAllItemIDs = sc.broadcast(allItemIDs)

val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)

import org.apache.spark.SparkContext

def predictMostListened(
      sc: SparkContext,
      train: RDD[Rating])(allData: RDD[(Int,Int)]) = {
            val bListenCount = sc.broadcast(
                  train.map(r => (r.product, r.rating)).
                  reduceByKey(_ + _).collectAsMap()
            )
            allData.map { case (user, product) =>
               Rating( user, product, bListenCount.value.getOrElse(product, 0.0) )
            }
}

val auc = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))

val evaluations = for (rank <- Array(10, 50);
                       lambda <- Array(1.0, 0.0001);
                       alpha <- Array(1.0, 40.0))
                  yield {
                       val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
                       val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
                       ((rank, lambda, alpha), auc)
                  }

evaluations.sortBy(_._2).reverse.foreach(println)


val recommendations = model.recommendProducts(2093760, 5)
recommendations.foreach(println)

val recommendedProductIDs = recommendations.map(_.product).toSet
artistByID.filter { case (id, name) => recommendedProductIDs.contains(id)}.
           values.collect().foreach(println)

rawArtistData.filter(line => line.contains("[unknown]") ).foreach(println)
rawUserArtistData.map(_.split(' ')(1).toInt).filter( _ == 1034635 ).count


val someUsers = allData.map(_.user).distinct().take(10)

val someRecommendations = someUsers.map(userID => model.recommendProducts(userID, 5))

someRecommendations.map(
     recs => recs.head.user + " -> " + recs.map(_.product).mkString(", ")
).foreach(println)

