#!/bin/bash

spark-shell \
	--conf spark.driver.memory=10g \
	--conf spark.executor.memory=10g \
	--conf spark.serializer=org.apache.spark.serializer.KryoSerializer
