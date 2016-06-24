#!/bin/sh

# set your 3rdparty_path
_3RDPARTH_PATH=..................

sudo cp $_3RDPARTH_PATH/bin/protoc /usr/bin
sudo chmod +x bin/gen_proto.sh
sudo ./bin/gen_proto.sh
mkdir build
cd build
cmake ..
make install -j8