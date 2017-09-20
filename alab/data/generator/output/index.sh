#!/bin/sh
if [ -z $1 ]; then
	exit
fi


DIR=$1

if [ -d $DIR ]; then
	echo split -l 5000 $DIR/index.txt $DIR/idx_
	split -l 5000 $DIR/index.txt $DIR/idx_
	echo find $DIR/ -name idx* -exec curl -XPOST localhost:9200/_bulk --data-binary @{} \;
	find $DIR/ -name idx* -exec curl -XPOST localhost:9200/_bulk --data-binary @{} \;
	echo rm $DIR/idx_*
	rm $DIR/idx_*
fi
