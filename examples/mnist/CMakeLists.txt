add_executable(examples_mnist
        mnist_example.cpp
)

target_link_libraries(examples_mnist
        PRIVATE
        tensor
        stdc++exp
)

target_include_directories(examples_mnist
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)