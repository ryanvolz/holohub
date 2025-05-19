# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
# Files to include in our module
list(APPEND rf_array_digital_metadata_python_files
    digital_metadata.py
)

# Define paths to copy in the holohub python module
list(TRANSFORM rf_array_digital_metadata_python_files
    PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/
    OUTPUT_VARIABLE rf_array_digital_metadata_python_src_files
)
set(rf_array_python_out_dir ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/rf_array)
file(MAKE_DIRECTORY ${rf_array_python_out_dir})
list(TRANSFORM rf_array_digital_metadata_python_files
    PREPEND ${rf_array_python_out_dir}/
    OUTPUT_VARIABLE rf_array_digital_metadata_python_out_files
)

# Target to copy
add_custom_target(rf_array_digital_metadata ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${rf_array_digital_metadata_python_src_files} ${rf_array_python_out_dir}
    DEPENDS ${rf_array_digital_metadata_python_src_files}
    BYPRODUCTS ${rf_array_digital_metadata_python_out_files}
)

add_custom_target(rf_array_digital_metadata_python_params ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/params ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/rf_array/params
)
