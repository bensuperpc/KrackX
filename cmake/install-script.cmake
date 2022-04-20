file(
    RELATIVE_PATH relative_path
    "/${krackx_INSTALL_CMAKEDIR}"
    "/${CMAKE_INSTALL_BINDIR}/${krackx_NAME}"
)

get_filename_component(prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
set(config_dir "${prefix}/${krackx_INSTALL_CMAKEDIR}")
set(config_file "${config_dir}/krackxConfig.cmake")

message(STATUS "Installing: ${config_file}")
file(WRITE "${config_file}" "\
get_filename_component(
    _krackx_executable
    \"\${CMAKE_CURRENT_LIST_DIR}/${relative_path}\"
    ABSOLUTE
)
set(
    KRACKX_EXECUTABLE \"\${_krackx_executable}\"
    CACHE FILEPATH \"Path to the krackx executable\"
)
")
list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${config_file}")
