include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package krackx)

install(
    TARGETS krackx_exe
    RUNTIME COMPONENT krackx_Runtime
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    krackx_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(krackx_INSTALL_CMAKEDIR)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${krackx_INSTALL_CMAKEDIR}"
    COMPONENT krackx_Development
)

# Export variables for the install script to use
install(CODE "
set(krackx_NAME [[$<TARGET_FILE_NAME:krackx_exe>]])
set(krackx_INSTALL_CMAKEDIR [[${krackx_INSTALL_CMAKEDIR}]])
set(CMAKE_INSTALL_BINDIR [[${CMAKE_INSTALL_BINDIR}]])
" COMPONENT krackx_Development)

install(
    SCRIPT cmake/install-script.cmake
    COMPONENT krackx_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
