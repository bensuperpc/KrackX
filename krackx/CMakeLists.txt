cmake_minimum_required(VERSION 3.16)

project(krackx VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.2 COMPONENTS Quick Core Qml Charts WebEngineQuick REQUIRED)

find_package(OpenMP 4.0)

qt_add_executable(appkrackx
    main.cpp
    counter.cpp counter.h
    about_compilation.cpp about_compilation.h
    applicationui.cpp applicationui.h
    chartdatamodel.h chartdatamodel.cpp
)

qt_add_qml_module(appkrackx
    URI krackx
    VERSION 1.0
    QML_FILES main.qml AboutPage.qml KrackPasswordPage.qml mainPage.qml SettingsPage.qml counter.qml Rectangle_test.qml
)

set_target_properties(appkrackx PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_compile_definitions(appkrackx
    PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(appkrackx
      PRIVATE Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::WebEngineQuick OpenMP::OpenMP_CXX)
else()
    target_link_libraries(appkrackx
        PRIVATE Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::WebEngineQuick)
endif()
