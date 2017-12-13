curl -s -L http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
  | bzip2 -cd \
  | hadoop fs -put - /user/ds/wikidump.xml
