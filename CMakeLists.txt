cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    krackx
    VERSION 0.1.0
    DESCRIPTION "Krack something"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

set(CMAKE_AUTOMOC ON)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

find_package(Qt6 6.2 COMPONENTS Quick Core Qml Charts Multimedia REQUIRED)
find_package(Qt6 6.2 COMPONENTS WebEngineQuick)
find_package(OpenMP 4.0)

add_library(
    krackx_lib OBJECT
    source/gta_sa.h source/gta_sa.cpp
)

target_include_directories(
    krackx_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

# set_target_properties(krackx_lib PROPERTIES AUTOMOC OFF)

target_compile_features(krackx_lib PUBLIC cxx_std_17)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(krackx_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Declare executable ----

qt_add_executable(krackx_exe 
    source/main.cpp
    source/counter.h source/counter.cpp
    source/about_compilation.h source/about_compilation.cpp
    source/applicationui.h source/applicationui.cpp
    source/chartdatamodel.h source/chartdatamodel.cpp
    source/liveimage.h source/liveimage.cpp
    source/imageprovider.h source/imageprovider.cpp
    source/gta_sa_ui.h source/gta_sa_ui.cpp
    source/TableModel.h
    source/utils.h
)

qt_add_qml_module(krackx_exe
    URI qml_files
    VERSION 1.0
    QML_FILES source/main.qml source/AboutPage.qml source/KrackPasswordPage.qml source/mainPage.qml source/SettingsPage.qml source/counter.qml source/GTA_SA.qml
)

add_executable(krackx::exe ALIAS krackx_exe)

# qt_add_executable

set_target_properties(
    krackx_exe PROPERTIES
    OUTPUT_NAME krackx
    EXPORT_NAME exe
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_compile_features(krackx_exe PRIVATE cxx_std_17)


# Qt6::WebEngineQuic
target_compile_definitions(krackx_exe
    PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)

target_link_libraries(krackx_exe PRIVATE krackx_lib)

target_link_libraries(krackx_exe PRIVATE Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::Multimedia)
    
if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(krackx_exe PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT krackx_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of krackx"
  )
endif()

include(cmake/dev-mode.cmake)
