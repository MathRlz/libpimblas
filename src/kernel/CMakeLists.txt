
set(CMAKE_C_COMPILER "${UPH}/bin/dpu-upmem-dpurte-clang")
set(CMAKE_CXX_COMPILER "${UPH}/bin/dpu-upmem-dpurte-clang")


set(common_kernel_libname "commonkernel")
add_subdirectory(common)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PIMBLAS_DEFAULT_KERENEL_DIR})

file(GLOB c_files "${CMAKE_CURRENT_LIST_DIR}/*.c")

list(REMOVE_ITEM c_files "${CMAKE_CURRENT_LIST_DIR}/clever_dot_product_uint32_8B.c")

foreach(file ${c_files})
message(${file})

get_filename_component(FILE_NAME_WE ${file} NAME_WE)
set(TNAME "${FILE_NAME_WE}.kernel")
add_executable(${TNAME} ${file})
target_compile_definitions(${TNAME} PRIVATE NR_TASKLETS=${NR_TASKLETS})

# required for hack dpurte-clang
target_link_options(${TNAME} PRIVATE -DNR_TASKLETS=${NR_TASKLETS}) 

target_include_directories(${TNAME} PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../share ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(${TNAME} PRIVATE ${common_kernel_libname})

install(TARGETS ${TNAME} DESTINATION kernel)



endforeach()



