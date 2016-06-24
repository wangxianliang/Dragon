FILE(GLOB_RECURSE TARGET_FILES *.c *.cpp *.cu *.cc)
FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${TARGET_FILES} ${SRC_FILES})

# ---[ link cuda
TARGET_LINK_LIBRARIES(${PROJECT_NAME}  ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})

FILE(GLOB targets ${3RDPARTY_LIBS}/*.so ${3RDPARTY_LIBS}/*.a ${3RDPARTY_LIBS}/*.lib) 
foreach(target ${targets})
	TARGET_LINK_LIBRARIES(${PROJECT_NAME}  ${target})
endforeach()

# ---[ link others
if(OS_LINUX)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} lmdb pthread)
endif()
if(OS_WINDOWS)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME}  shlwapi.lib)
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME dragon)

# ---[ install
install (TARGETS ${PROJECT_NAME} DESTINATION bin)

