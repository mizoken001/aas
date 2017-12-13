
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val rawData = sc.textFile("hdfs:///user/ds/ch04/covtype.data",8)

// A First Decision Tree

// LabeledPoint Parse
val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init) 
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
trainData.cache()
cvData.cache()
testData.cache()


// DecisionTree Classifier Model make
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

def getMetrics( model: DecisionTreeModel, data: RDD[LabeledPoint] ):MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
               (model.predict(example.features), example.label)
        )
    new MulticlassMetrics(predictionsAndLabels)
}

val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)
println("Learned classification tree model:\n" + model.toDebugString)



// DecisionTree Classifier Model evaluation
val metrics = getMetrics(model, cvData)

metrics.confusionMatrix
metrics.precision

(0 until 7).map( cat => (metrics.precision(cat), metrics.recall(cat)) ).
            foreach(println)


import org.apache.spark.rdd._

def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)

trainPriorProbabilities.zip(cvPriorProbabilities).map {
   case (trainProb, cvProb) => trainProb * cvProb
}.sum

(0 until 7).map( cat => (metrics.precision(cat), metrics.recall(cat)) ).
            foreach(println)


// Tuning Decision Trees
// Decision Tree Hyperparameters

val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth <- Array(1, 20);
       bins <- Array(10, 300))
  yield {
       val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), impurity, depth, bins)
       val predictionsAndLabels = cvData.map(example =>
               (model.predict(example.features), example.label)
           )
       val accuracy = new MulticlassMetrics(predictionsAndLabels).precision
       ((impurity, depth, bins), accuracy)
}
evaluations.sortBy(_._2).reverse.foreach(println)


val model = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)
val metrics = getMetrics(model, testData)

metrics.confusionMatrix
metrics.precision

val metrics = getMetrics(model, cvData)
metrics.precision


// Categorical Features Revisited

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val wilderness = values.slice(10, 14).indexOf(1.0).toDouble 
  val soil = values.slice(14, 54).indexOf(1.0).toDouble
  val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}
val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
trainData.cache()
cvData.cache()
testData.cache()

val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth <- Array(10, 20, 30);
       bins <- Array(40, 300))
  yield {
       val model = DecisionTree.trainClassifier(
	      trainData, 7, Map(10 -> 4, 11 -> 40),
              impurity, depth, bins)
       val trainAccuracy = getMetrics(model, trainData).precision
       val cvAccuracy = getMetrics(model, cvData).precision
       ((impurity, depth, bins), (cvAccuracy,trainAccuracy))
}
evaluations.sortBy(_._2).map{ case (x,(y,z)) => (x,(z,y)) }.reverse.foreach(println)

val model = DecisionTree.trainClassifier(trainData.union(cvData), 7,
             Map(10 -> 4, 11 -> 40), "entropy", 30, 300)

val metrics = getMetrics(model, testData)

metrics.precision
metrics.confusionMatrix

(0 until 7).map( cat => (metrics.precision(cat), metrics.recall(cat)) ).
            foreach(println)


// Random Decision Forests

import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._


val forest = RandomForest.trainClassifier(trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)

def getMetrics( model: RandomForestModel, data: RDD[LabeledPoint] ):MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
               (model.predict(example.features), example.label)
        )
    new MulticlassMetrics(predictionsAndLabels)
}
val metrics = getMetrics(forest, testData)

metrics.precision
metrics.confusionMatrix

val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth <- Array(10, 20, 30);
       bins <- Array(40, 300))
  yield {
       val forest = RandomForest.trainClassifier(
              trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto",
              impurity, depth, bins)
       val trainAccuracy = getMetrics(forest, trainData).precision
       val cvAccuracy = getMetrics(forest, cvData).precision
       ((impurity, depth, bins), (cvAccuracy,trainAccuracy))
}
evaluations.sortBy(_._2).map{ case (x,(y,z)) => (x,(z,y)) }.reverse.foreach(println)


(0 until 7).map( cat => (metrics.precision(cat), metrics.recall(cat)) ).
            foreach(println)


val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
val vector = Vectors.dense(input.split(',').map(_.toDouble))
forest.predict(vector)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Model Test Error = " + testErr)

// Evaluate forest on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = forest.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Forest Model Test Error = " + testErr)
