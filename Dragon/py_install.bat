@echo off
set _3RDPARTY_PATH=..................
set ANACONDA_PATH=..................

copy /y .\lib\_dragon.dll .\ext\python\dragon\_dragon.pyd
copy /y %_3RDPARTY_PATH%\cv2.pyd %ANACONDA_PATH%\lib\cv2.pyd
set SRC_DIR=.\src\protos
set DST_DIR=.\ext\python\dragon
set PROTO_NAME=dragon
echo Check Source Proto Path:  %SRC_DIR%
echo Check Destination Proto Path:  %DST_DIR%
echo Check Proto Files Name :  %PROTO_NAME%.proto
echo Protocol Buffer£ºCompliing for dragon.proto.....
start protoc -I=%SRC_DIR% --python_out=%DST_DIR% %SRC_DIR%\%PROTO_NAME%.proto
echo Protocol Buffer£ºCompliing complete!
xcopy /s /y .\ext\python\dragon %ANACONDA_PATH%\lib\site-packages\dragon\
pause