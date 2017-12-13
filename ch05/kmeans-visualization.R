install.packages("rgl")

clusters_data <- read.csv(pipe("hdfs dfs -cat /user/ds/ch05/sample/*"))
clusters <- clusters_data[1]
data <- data.matrix(clusters_data[-c(1)])
rm(clusters_data)

random_projection <- matrix(data = rnorm(3*ncol(data)), ncol = 3)
random_projection_norm <-
  random_projection /
    sqrt(rowSums(random_projection*random_projection))

projected_data <- data.frame(data %*% random_projection_norm)

library(rgl)

num_clusters <- nrow(unique(clusters))
palette <- rainbow(num_clusters)
colors = sapply(clusters, function(c) palette[c])
plot3d(projected_data, col = colors, size = 10) 
