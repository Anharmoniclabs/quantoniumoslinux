set(TARGET llama-mimosa-kv-test)
add_executable(${TARGET} mimosa_kv_test.cpp)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
