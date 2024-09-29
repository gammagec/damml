add_library(tensor
        tensor.cpp
)

target_include_directories(tensor
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(tensor_test
        tensor_test.cpp
)

target_include_directories(tensor_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(tensor_test
        PRIVATE
        tensor
        stdc++exp
        ${Boost_LIBRARIES}
)

add_test(NAME TensorTest COMMAND tensor_test)