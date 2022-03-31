import QtQuick 2.13
import QtQuick.Window 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.13

Page {
    id: about_page
    title: qsTr("About page")
    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 5
        GroupBox {
            title: qsTr("About this app")
            Column {
                Text {
                    text: qsTr("Created by: Bensuperpc")
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                    wrapMode: Text.WordWrap
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
                        acceptedButtons: Qt.NoButton // we don't want to eat clicks on the Text
                        cursorShape: parent.hoveredLink ? Qt.PointingHandCursor : Qt.ArrowCursor
                    }
                }
            }
        }

        GroupBox {
            title: qsTr("App and Compiler")
            Column {
                Text {
                    text: qsTr("Compiler :" + about_compilation.return_Compiler_name(
                                   ))
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                }
                Text {
                    text: qsTr("Compiler vers:" + about_compilation.return_Compiler_version(
                                   ))
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                }
                Text {
                    text: qsTr("C++ version :" + about_compilation.return_Cplusplus_used(
                                   ))
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                }
                Text {
                    text: qsTr(
                              "Build date :" + about_compilation.return_BuildDate(
                                  ) + " : " + about_compilation.return_BuildTime(
                                  ))
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                }
            }
        }
        GroupBox {
            title: qsTr("Qt")
            Column {
                Text {
                    text: qsTr("Qt version :" + qtversion)
                    color: "white"
                    font.bold: true
                    font.pointSize: 10
                }
            }
        }

        Image {
            fillMode: Image.PreserveAspectFit
            source: "qrc:images/qt-logo@4x.png"
            MouseArea {
                onClicked: {

                }
            }
        }
        //https://stackoverflow.com/questions/23667088/qtquick-dynamic-images-and-c/
    }
}
