import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material
import MyApp.Images

Page {
    title: qsTr("BruteForce GTA SA")
    id: page

    Flickable {
        anchors.fill: parent
        contentHeight: columnLayout.implicitHeight
        contentWidth: columnLayout.implicitWidth

        flickableDirection: Flickable.AutoFlickIfNeeded
        ColumnLayout {
            id: columnLayout
            spacing: 10
            width: page.width
            height: children.height

            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                GroupBox {
                    title: qsTr("Image")
                    Layout.alignment: Qt.AlignHCenter
                    ColumnLayout {
                        RowLayout {
                            Image {
                                source: "image: //colors/red"
                            }
                            Image {
                                source: "image: //colors/white"
                            }
                            Image {
                                source: "image: //colors/blue"
                            }
                        }
                    }
                }
            }


            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                GroupBox {
                    title: qsTr("Image")
                    Layout.alignment: Qt.AlignHCenter
                    ColumnLayout {
                        RowLayout {
                            Rectangle {
                                width: 480
                                height: 480
                                color: "transparent"
                                LiveImage {
                                    anchors.fill: parent
                                    // width: 480
                                    // height: 480
                                    enable_rescale: true
                                    image: LiveImageProvider.image
                                }
                            }
                        }
                    }
                }
            }
        }

        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
