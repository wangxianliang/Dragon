# Dragon: A Light Deep Learning Framework

## Specifics !
- CaffeModels Support

- Distributed Architectures Support
 	- Device Parallelism: Use muti-GPUs to run a huge Neural Network(e.g. Res-Net)
	- Data Parallelism: Update several Neural Networks with a ParameterServer
	- Python Parallelism: Be compatible with Python

- Symbols Support(in development)

- Cross Platforms(Linux/Windows) and United Deployment

## How to Install ?
### Requirements
1. CUDA
2. Anaconda [Optional]
3. OpenMPI [Optional]
4. OpenCV [Optional]

### Installation 
1. clone this repository

2. download and install [CUDA](https://developer.nvidia.com/cuda-toolkit)
	- we recommend install CUDA8 or higher to support GCC5

3. download 3rdparty
	- [Win64_MSVC_Release](http://pan.baidu.com/s/1kVIY9eB)
	- Linux64_GCC4
	- [Linux64_GCC5](http://pan.baidu.com/s/1dFHiVhZ)

4. install [Anaconda](https://www.continuum.io/downloads) with x64-py2.7  [Optional]
	- allow importing environment variable into bashrc
	- install protobuf
	```Shell
	pip install protobuf
	```

5. configure Dragon/CMakeLists.txt
	- select platforms
	- select optional libraries
	- set 3rdparty path
	- set python path  [Optional]
	- set cuda compiling architectures if necessary

6. set environment variables
	- Linux:
		- create dragon.conf
		```Shell
		sudo gedit /etc/ld.so.conf.d/dragon.conf
		```
		- write down your 3rdparty/lib path at the first line
		 	- e.g. /home/xxx/3rdparty/lib
		- rebuild the scaning cache
		```Shell 
		sudo ldconfig
		```
	- Windows
		- add 3rdparty/bin to system environment variables
			- e.g PATH=........;C:\3rdparty\bin;

7. install MPI [Optional]
	- Linux:
		- we use OpenMPI which support "cuda-aware-mpi"
		- see more: 
			- https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
			- https://www.open-mpi.org/faq/?category=buildcuda
		- configure 3rdparty/setup_mpi.sh
			- disable cuda-aware support if necessary
		- run 3rdparty/setup_mpi.sh
		```Shell
		sudo chmod +x setup_mpi.sh
		sudo ./setup_mpi.sh
		```
	- Windows:
		- we use Microsoft MPI which can perfectly run at lastest Windows10
		- Microsoft MPI is intergrated into 3rdparty and you should do nothing

8. compile
	- Linux: 
		- install cmake
		```Shell
  		sudo apt-get install cmake
  		```
  		- configure Dragon/main_install.sh
			- set 3rdparty path
		- run Dragon/main_install.sh
		```Shell
		sudo chmod +x main_install.sh
		sudo ./main_install.sh
		```
	- Windows:
		- install cmake-gui
		- mkdir Dragon/build
		- configure and generate MSVC project in Dragon/build
		- open Dragon/build/Dragon.sln
		- compile and generate for "INSTALL" solution

9. install PyDragon [Optional]
	- Linux:
		- configure Dragon/py_install.sh
			- set anaconda path
		- run Dragon/py_install.sh
		```Shell
		sudo chmod +x py_install.sh
  		sudo ./py_install.sh
  		```
	- Windows:
		- configure Dragon/py_install.bat
			- set 3rdparty/bin path
			- set anaconda path
		- run Dragon/py_install.bat

## Hints

Developer: PanTing, HeFei University of Technology at Xuancheng

We will change the codes frequently

This reop is also regarded as the Author's Graduation Project

Thanks to Caffe & Theano

## License and Citation

Dragon is a distribution of the Caffe.

Please cite Caffe firstly in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, 
      Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
