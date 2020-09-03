cmake_minimum_required(VERSION 3.15)
project(tfop)

#SET(CMAKE_C_COMPILER "/usr/bin/gcc")
#SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 14)


###############################################################################
## file glob/variables ########################################################
###############################################################################

# these instructions search the directory tree when cmake is
# invoked and put all files that match the pattern in the variables
# `sources` and `data`
file(GLOB_RECURSE sources src/main/ops/*.cpp src/main/ops/*.h
        src/main/kernels/*.cpp src/main/kernels/*.h)
file(GLOB_RECURSE sources_test src/test/*.cpp)
file(GLOB_RECURSE data resources/*)
# you can use set(sources src/main.cpp) etc if you don't want to
# use globing to find files automatically
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")


###############################################################################
## target definitions #########################################################
###############################################################################

# add the data to the target, so it becomes visible in some IDE
add_library(tfop SHARED ${sources} ${data})

# just for example add some compiler flags
target_compile_options(tfop PUBLIC -std=c++1y -Wall -Wfloat-conversion -D_GLIBCXX_USE_CXX11_ABI=0)

# this lets me include files relative to the root src dir with a <> pair
target_include_directories(tfop PUBLIC src/main)


###############################################################################
## dependencies ###############################################################
###############################################################################

# 相当于g++命令的-I参数，也相当于加路径到CPLUS_INCLUDE_PATH环境变量
# include_directories(include_path1 include_path2 ...)
# include_directories("/Users/bytedance/.conda/envs/sail/lib/python2.7/site-packages/tensorflow/include")

# 用于在路径中查找一个名为name的(头)文件, 如果找到, 则将全路径名保存在<VAR>中, 可与include_directories一起使用
# 但是find_path没有递归能力, 不会在子文件夹下找
# find_path(<VAR> NAMES name PATHS path1 path2 ...)


# 相当于g++命令的-L参数，也相当于加路径到LD_LIBRARY_PATH环境变量
# link_directories(lib_path1 lib_path1 ...)
# link_directories("/Users/bytedance/.conda/envs/sail/lib/python2.7/site-packages/tensorflow")

# 设置要链接的库文件的名称, 一般配合link_directories一起使用, 相当于g++命令的-l参数, 多用于引入动态链接库
# 它会在lib_path中自动找lib, 然后加到target编译的命令行, 所以只能用在add_executable/add_library之后
# {lib_name}, -l{lib_name} lib{lib_name}.so lib{lib_name}.a 都是合法的
# target_link_libraries(myProject lib1 lib2 ...)
# target_link_libraries(tfop -ltensorflow_framework)

# 添加需要链接的库文件(全路径名), 相当于g++命令的-l参数, 多用于引入静态链接库
# 用在add_executable/add_library之前, 所以会加到所有的target中, 但它不会自动找路径
# link_libraries("lib_path1/libeng.so"　"lib_path2/libmx.so")

# 在多个路径中查找一个名为name的库文件, 如果找到, 则将全路径名保存在<VAR>中, 可与link_libraries一起使用
# find_library(<VAR> NAMES name PATHS lib_path1, lib_path2 ...)


# 用于一键加载引入package所要的INCLUDE/LIBRARY, 下面是一个例子
find_package(TensorFlow REQUIRED)
if(TensorFlow_FOUND)
    target_include_directories(tfop PRIVATE "${TensorFlow_INCLUDE_DIR}")
    target_link_libraries(tfop "${TensorFlow_LIBRARY}")
else(TensorFlow_FOUND)
    message(FATAL_ERROR "TensorFlow library not found")
endif(TensorFlow_FOUND)

# target_link_libraries(example PUBLIC ${Boost_LIBRARIES})


