add_executable(linear_test
        linear_test.cpp
)

target_include_directories(linear_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(linear_test
        PRIVATE
        tensor
        stdc++exp
        ${Boost_LIBRARIES}
)

add_test(NAME LinearTest COMMAND linear_test)
