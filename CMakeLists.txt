cmake_minimum_required(VERSION 3.10.1)
project(test_neon)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -o3 -ftree-vectorize -mtune=cortex-a72")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}
) # ref: https://keunjun.blog/2018/10/07/cmake-10-%EB%AA%A8%EB%93%88/


find_package(cmake_modules)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)
set(SRC ${SRC}
    src/utility/timer.cpp
)

include_directories( ${includeDirs} )
add_executable (test_add_u8 src/test_add_u8.cpp ${SRC})
add_executable (test_add_u16 src/test_add_u16.cpp ${SRC})

add_executable (test_muladd_u8 src/test_add_u8.cpp ${SRC})
add_executable (test_muladd_u16 src/test_muladd_u16.cpp ${SRC})