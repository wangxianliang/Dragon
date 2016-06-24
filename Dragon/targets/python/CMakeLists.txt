FILE(GLOB_RECURSE TARGET_FILES *.c *.cpp *.cu *.cc)
FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)
CUDA_ADD_LIBRARY(${PROJECT_NAME}Python SHARED ${TARGET_FILES} ${SRC_FILES})

# ---[ link cuda
TARGET_LINK_LIBRARIES(${PROJECT_NAME}Python  ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})

FILE(GLOB targets ${3RDPARTY_LIBS}/*.so ${3RDPARTY_LIBS}/*.a ${3RDPARTY_LIBS}/*.lib) 
foreach(target ${targets})
IF(NOT (  (${target} STREQUAL ${3RDPARTY_LIBS}/libgflags.a)  OR (${target} STREQUAL ${3RDPARTY_LIBS}/gflags.lib)  ) )
	TARGET_LINK_LIBRARIES(${PROJECT_NAME}Python  ${target})
ENDIF()
endforeach()


# ---[ link others
if(OS_LINUX)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME}Python lmdb pthread)
endif()
if(OS_WINDOWS)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME}Python  shlwapi.lib)
endif()

set_target_properties(${PROJECT_NAME}Python PROPERTIES OUTPUT_NAME _dragon)

# ---[ install
install (TARGETS ${PROJECT_NAME}Python DESTINATION lib)

