# ---------------- Welcom To Use Dragon  ----------------

PROJECT(Dragon)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

# ---------------- Welcom To Use Dragon   ----------------
# ---------------- User Config ----------------


# set OS
set(OS_LINUX ON)
set(OS_WINDOWS OFF)

# set optional libraries
set(WITH_PYTHON ON)
set(WITH_MPI ON)
set(WITH_OPENCV OFF)

# set your 3rdparty
set(3RDPARTY_DIR ..................)

# set if you want to build pydragon
set(ANACONDA_DIR ..................)
set(NUMPY_DIR ..................)

# set CUDA compiling architecture
set(CUDA_ARCH -gencode arch=compute_20,code=sm_20
	      -gencode arch=compute_30,code=sm_30
              -gencode arch=compute_35,code=sm_35
	      -gencode arch=compute_50,code=sm_50)

# ---------------- User Config ----------------



# ---------------- Do Not Edit Following Items ----------------


#                           _ooOoo_
#                          o8888888o
#                          88" . "88
#                          (| -_- |)
#                           O\ = /O
#                       ____/`---'\____
#                      .   ' \\| |// `.
#                     / \\||| : |||// \
#                     / _||||| -:- |||||- \
#                      | | \\\ - /// | |
#                     | \_| ''\---/'' | |
#                     \ .-\__ `-` ___/-. /
#                  ___`. .' /--.--\ `. . __
#               ."" '< `.___\_<|>_/___.' >'"".
#               | | : `- \`.;`\ _ /`;.`/ - ` : | |
#                \ \ `-. \_ __\ /__ _/ .-` / /
#         ======`-.____`-.___\_____/___.-`____.-'======
#                           `=---='
#


# ---------------- If You Are Not Goot At CMake ----------------


# ---[ Dependencies
FIND_PACKAGE(CUDA REQUIRED)


# ---[ Config types
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)

# ---[ Includes
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})
include_directories(${3RDPARTY_DIR}/include)
include_directories(${3RDPARTY_DIR}/include/mpi)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${NUMPY_DIR}/core/include)
include_directories(${ANACONDA_DIR}/include/python2.7)
include_directories(${ANACONDA_DIR}/include)

# ---[ libs
set(3RDPARTY_LIBS ${3RDPARTY_DIR}/lib)
link_directories(${3RDPARTY_LIBS})

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)

# ---[ defines
if (NOT WITH_PYTHON) 
	ADD_DEFINITIONS(-DNO_PYTHON)
endif()
if (NOT WITH_MPI) 
	ADD_DEFINITIONS(-DNO_MPI)
endif()
if (NOT WITH_OPNECV) 
	ADD_DEFINITIONS(-DNO_OPENCV)
endif()
if (OS_WINDOWS) 
	ADD_DEFINITIONS(-D_WINDOWS_MSVC_)
endif()


# ---[ Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
if (OS_LINUX) 
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -m64 -fpermissive")
endif()
# ---[ Warnings

# ---[ Subdirectories
add_subdirectory(targets/exec)
if(WITH_PYTHON)
	add_subdirectory(targets/python)
endif()








