cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    krackx
    VERSION 0.1.0
    DESCRIPTION "Krack something"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

set(CMAKE_AUTOMOC ON)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

find_package(Qt6 6.2 COMPONENTS Quick Core Qml Charts Multimedia REQUIRED)
# find_package (Threads REQUIRED)

# find_package(Qt6 6.2 COMPONENTS WebEngineQuick)
find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

# Download ThreadPool
include(FetchContent)
FetchContent_Declare(
    thread-pool
    FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps
    URL      https://github.com/bshoshany/thread-pool/archive/refs/tags/v2.0.0.zip
    URL_HASH SHA512=c59114dfc64f1e0837780c4b5d6815dcaeeeeae99388799d9375826cbde24acabb576922228fa7bf7c9a334d434eb8727e2deb45e289c944278d153617877e8a
)
FetchContent_MakeAvailable(thread-pool)

if (CUDAToolkit_FOUND)
  add_library(
      krackx_lib OBJECT
      source/gta_sa.hpp 
      source/gta_sa.cpp
      source/cuda/kernel/kernel.cu
      source/cuda/kernel/kernel.cuhpp
      source/cuda/kernel.hpp
      source/cuda/kernel.cpp
  )
else()
  add_library(
      krackx_lib OBJECT
      source/gta_sa.hpp 
      source/gta_sa.cpp
  )
endif()

target_include_directories(
    krackx_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

# set_target_properties(krackx_lib PROPERTIES AUTOMOC OFF)
if (CUDAToolkit_FOUND)
  target_include_directories(krackx_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(krackx_lib PUBLIC CUDA::cudart)

  target_include_directories(krackx_lib PUBLIC "${PROJECT_SOURCE_DIR}/source/cuda_kernel")
endif()

target_include_directories(krackx_lib PUBLIC "${thread-pool_SOURCE_DIR}")

target_compile_features(krackx_lib PUBLIC cxx_std_17)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(krackx_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Declare executable ----

# qt_add_executable
add_executable(krackx_exe 
    source/main.cpp
    source/counter.h source/counter.cpp
    source/about_compilation.h source/about_compilation.cpp
    source/applicationui.h source/applicationui.cpp
    source/chartdatamodel.h source/chartdatamodel.cpp
    source/liveimage.h source/liveimage.cpp
    source/imageprovider.h source/imageprovider.cpp
    source/gta_sa_ui.h source/gta_sa_ui.cpp
    source/TableModel.h source/TableModel.cpp
    source/utils.h
    source/compilation.hpp source/compilation.cpp
)


#set_source_files_properties(source/qml/main.qml  PROPERTIES
#    QT_RESOURCE_ALIAS main.qml
#)

qt_add_qml_module(krackx_exe
    URI qml_files
    VERSION 1.0
    RESOURCE_PREFIX /bensuperpc.com
    QML_FILES 
    source/qml/main.qml 
    source/qml/AboutPage.qml 
    source/qml/KrackPasswordPage.qml 
    source/qml/mainPage.qml 
    source/qml/SettingsPage.qml 
    source/qml/counter.qml 
    source/qml/image_test.qml 
    source/qml/GTA_SA.qml
    RESOURCES
    img/Profile_400x400.jpg
)

add_executable(krackx::exe ALIAS krackx_exe)

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

# target_link_libraries(krackx_exe PRIVATE Threads::Threads)
    
if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(krackx_exe PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Install rules ----

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/resource/Profile_400x400.jpg ${CMAKE_CURRENT_BINARY_DIR}/resource/Profile_400x400.jpg COPYONLY)

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
