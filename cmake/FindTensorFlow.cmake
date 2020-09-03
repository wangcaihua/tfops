# 在指定目录下寻找头文件和动态库文件的位置，可以指定多个目标路径(https://zhuanlan.zhihu.com/p/97369704)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE TensorFlow_INCLUDE_DIR)
string(REGEX REPLACE "\n$" "" TensorFlow_INCLUDE_DIR "${TensorFlow_INCLUDE_DIR}")

execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        OUTPUT_VARIABLE TF_Lib)
string(REGEX REPLACE "\n$" "" TF_Lib "${TF_Lib}")

find_library(TensorFlow_LIBRARY NAMES tensorflow_framework PATHS "${TF_Lib}")

if (TensorFlow_INCLUDE_DIR AND TensorFlow_LIBRARY)
    set(TensorFlow_FOUND TRUE)
endif (TensorFlow_INCLUDE_DIR AND TensorFlow_LIBRARY)
