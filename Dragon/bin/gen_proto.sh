#!/bin/sh

SRC_DIR=src/protos
DST_DIR=src/protos
PROTO_NAME=dragon
echo Check Source Proto Path:  $SRC_DIR
echo Check Destination Proto Path:  $DST_DIR
echo Check Proto Files Name :  $PROTO_NAME.proto
echo Protocol Buffer Compliing for dragon.proto.....
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/$PROTO_NAME.proto
echo Protocol Buffer Compliing complete!