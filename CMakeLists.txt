cmake_minimum_required(VERSION 3.10.1)
project(test_neon)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -o2 -ftree-vectorize")
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
add_executable (test_addition src/main_addition.cpp ${SRC})