add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        fused_copy_sfa_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedCopySfa
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    set(fused_copy_sfa_depends "attention/fused_copy_sfa" CACHE INTERNAL "")
    add_modules_sources(OPTYPE fused_copy_sfa ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_tail_attention/op_host
    )
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_tail_attention/op_host/sparse_tail_attention_tiling.cpp
    )
endif()
