val rawblocks = sc.textFile("hdfs:///user/ds/ch02/")

val head = rawblocks.take(10)

head.foreach(println)

def isHeader(line: String) = line.contains("id_1")

val noheader = rawblocks.filter(x => !isHeader(x))
val header = rawblocks.filter(x => isHeader(x))

def toDouble(s: String) = if ( s == "?" ) Double.NaN else s.toDouble

case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)

def parse(line: String) = {
         val pieces = line.split(',')
         val id1 = pieces(0).toInt
         val id2 = pieces(1).toInt
         val scores = pieces.slice(2, 11).map(toDouble)
         val matched = pieces(11).toBoolean
         MatchData(id1, id2, scores, matched)
       }

val mds = head.filter(x => !isHeader(x)).map(x => parse(x))

val parsed = noheader.map(line => parse(line))

parsed.cache()

val mds = parsed.take(10)
val mds_grouped = mds.groupBy(md => md.matched)
mds_grouped.mapValues(x => x.size).foreach(println)

val parsed_grouped = parsed.groupBy(md => md.matched)
parsed_grouped.mapValues(x => x.size).foreach(println)

val matched = parsed.map( x => ( x.matched, 1 ) ).reduceByKey( _ + _ )

val matchCounts = parsed.map(md => md.matched).countByValue()
val matchCountsSeq = matchCounts.toSeq

matchCountsSeq.sortBy(_._1).foreach(println)
matchCountsSeq.sortBy(_._2).foreach(println)
matchCountsSeq.sortBy(_._2).reverse.foreach(println)

import java.lang.Double.isNaN
val stats = (0 until 9).map(i => { parsed.map(md => md.scores(i)).filter(!isNaN(_)).stats() })

stats.foreach( println )

:load ch02/StatsWithMissing.scala

val nas1 = NAStatCounter(10.0)
val nas2 = NAStatCounter(Double.NaN)
nas1.add(2.1)
nas1.merge(nas2)

val arr = Array(1.0, Double.NaN, 17.29)
val nas = arr.map(d => NAStatCounter(d))

val nasRDD = parsed.map(md => { md.scores.map(d => NAStatCounter(d)) })

val reduced = nasRDD.reduce((n1, n2) => { n1.zip(n2).map { case (a, b) => a.merge(b) } })
reduced.foreach(println)

val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))

statsm.zip(statsn).map{ case(m, n) => (m.missing + n.missing, m.stats.mean - n.stats.mean) }.foreach(println)

def naz(d: Double) = if (Double.NaN.equals(d)) 0.0 else d

case class Scored(md: MatchData, score: Double)

val ct = parsed.map(md => {
  val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
  Scored(md, score)
})

val score4 = ct.filter(s => s.score >= 4.0)
score4.map(s => s.md.matched).countByValue()

val score2 = ct.filter(s => s.score >= 2.0)
score2.map(s => s.md.matched).countByValue()


