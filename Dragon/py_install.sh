#!/bin/sh

# set your anaconda_path
ANACONDA_PATH=..................

cp lib/lib_dragon.so ext/python/dragon/_dragon.so
SRC_DIR=./src/protos
DST_DIR=./ext/python/dragon
PROTO_NAME=dragon
echo Check Source Proto Path:  $SRC_DIR
echo Check Destination Proto Path:  $DST_DIR
echo Check Proto Files Name :  $PROTO_NAME.proto
echo Protocol Buffer Compliing for dragon.proto.....
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/$PROTO_NAME.proto
echo Protocol Buffer Compliing complete!
sudo cp -r ext/python/dragon $ANACONDA_PATH/lib/python2.7/site-packages