import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

Page {
    title: qsTr("About Page")
    id: page

    Flickable {
        anchors.fill: parent
        contentHeight: columnLayout.implicitHeight
        contentWidth: columnLayout.implicitWidth
        flickableDirection: Flickable.AutoFlickIfNeeded

        ColumnLayout {
            // unique child
            id: columnLayout
            spacing: 10
            width: page.width // ensure correct width
            height: children.height // ensure correct height

            GroupBox {
                title: qsTr("About this app")
                Layout.alignment: Qt.AlignHCenter
                Column {
                    Text {
                        text: qsTr("Created by: <a href='https://github.com/Bensuperpc'>Bensuperpc</a>")
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                        wrapMode: Text.WordWrap
                        onLinkActivated: Qt.openUrlExternally(link)

                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.NoButton
                            cursorShape: parent.hoveredLink ? Qt.PointingHandCursor : Qt.ArrowCursor
                        }
                    }
                    Text {
                        text: "Source: <a href='https://github.com/Bensuperpc/KrackX'>Click here</a>"
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                        wrapMode: Text.WordWrap
                        onLinkActivated: Qt.openUrlExternally(link)

                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.NoButton
                            cursorShape: parent.hoveredLink ? Qt.PointingHandCursor : Qt.ArrowCursor
                        }
                    }
                }
            }

            GroupBox {
                title: qsTr("App and Compiler")
                Layout.alignment: Qt.AlignHCenter
                Column {
                    Text {
                        text: qsTr("Compiler: " + about_compilation.return_Compiler_name(
                                       ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                    Text {
                        text: qsTr("Compiler vers: " + about_compilation.return_Compiler_version(
                                       ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                    Text {
                        text: qsTr("C++ version: " + about_compilation.return_Cplusplus_used(
                                       ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                    Text {
                        text: qsTr(
                                  "Build date: " + about_compilation.return_BuildDate(
                                      ) + " : " + about_compilation.return_BuildTime(
                                      ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                }
            }
            GroupBox {
                title: qsTr("Libs")
                Layout.alignment: Qt.AlignHCenter
                Column {
                    Text {
                        text: qsTr("Qt version: " + qtversion)
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }

                    Text {
                        text: qsTr(
                                  "OpenMP: " + about_compilation.openmpIsEnable(
                                      ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                    Text {
                        text: qsTr("Nvidia CUDA: " + about_compilation.cudaIsEnable(
                                       ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                    Text {
                        text: qsTr(
                                  "OpenCL: " + about_compilation.openclIsEnable(
                                      ))
                        color: "white"
                        font.bold: true
                        font.pointSize: 10
                    }
                }
            }


            /*
            Image {
                fillMode: Image.PreserveAspectFit
                source: "qrc:images/qt-logo@4x.png"
                MouseArea {
                    onClicked: {

                    }
                }
            }
            */
            //https://stackoverflow.com/questions/23667088/qtquick-dynamic-images-and-c/
        }

        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
