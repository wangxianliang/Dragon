@echo off
set SRC_DIR=..\src\protos
set DST_DIR=..\src\protos
set PROTO_NAME=dragon
echo Check Source Proto Path:  %SRC_DIR%
echo Check Destination Proto Path:  %DST_DIR%
echo Check Proto Files Name :  %PROTO_NAME%.proto
echo ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
echo Protocol Buffer£ºCompliing for dragon.proto.....
start protoc -I=%SRC_DIR% --cpp_out=%DST_DIR% %SRC_DIR%\%PROTO_NAME%.proto
echo Protocol Buffer£ºCompliing complete!
pause