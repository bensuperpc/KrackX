cmake_minimum_required(VERSION 3.16)

project(krackx VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.2 COMPONENTS Quick Core Qml Charts Multimedia REQUIRED)
find_package(Qt6 6.2 COMPONENTS WebEngineQuick)

find_package(OpenMP 4.0)

qt_add_executable(appkrackx
    main.cpp
    counter.h counter.cpp
    about_compilation.h about_compilation.cpp
    applicationui.h applicationui.cpp
    chartdatamodel.h chartdatamodel.cpp
    liveimage.h liveimage.cpp
    imageprovider.h imageprovider.cpp
    gta_sa.h gta_sa.cpp
    gta_sa_ui.h gta_sa_ui.cpp
    gta_sa.h gta_sa.cpp
    TableModel.h
    utils.h
)

qt_add_qml_module(appkrackx
    URI krackx
    VERSION 1.0
    QML_FILES main.qml AboutPage.qml KrackPasswordPage.qml mainPage.qml SettingsPage.qml counter.qml GTA_SA.qml
)

set_target_properties(appkrackx PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# Qt6::WebEngineQuic
target_compile_definitions(appkrackx
    PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(appkrackx
      PRIVATE Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::Multimedia OpenMP::OpenMP_CXX)
else()
    target_link_libraries(appkrackx
        PRIVATE Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts k Qt6::Multimedia)
endif()
