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
                    title: qsTr("BruteForce GTA SA")
                    ColumnLayout {
                        RowLayout {
                            TextField {
                                id: minRangeValue
                                placeholderText: qsTr("Enter minimal range value")
                                text: gta_sa.minRangeValue.toLocaleString(
                                'fullwide', {
                                "useGrouping": false
                            })
                            selectByMouse: true
                            validator: RegularExpressionValidator {
                            regularExpression: /[0-9]+/
                        }
                    }

                    Binding {
                        target: gta_sa
                        property: "minRangeValue"
                        value: minRangeValue.text
                    }
                }
                RowLayout {
                    TextField {
                        id: maxRangeValue
                        placeholderText: qsTr("Enter maximum range value")
                        text: gta_sa.maxRangeValue.toLocaleString(
                        'fullwide', {
                        "useGrouping": false
                    })
                    selectByMouse: true
                    validator: RegularExpressionValidator {
                    regularExpression: /[0-9]+/
                }
            }
            Binding {
                target: gta_sa
                property: "maxRangeValue"
                value: maxRangeValue.text
            }
        }
        RowLayout {
            Button {
                id: launchRunner
                Layout.alignment: Qt.AlignHCenter
                // text: qsTr("Launch Bruteforce")
                text: gta_sa.buttonValue
                onClicked: gta_sa.runOp()
            }
        }
    }
}
GroupBox {
    title: qsTr("Settings")
    ColumnLayout {
        RowLayout {
            CheckBox {
                id: enableOpenMP
                text: qsTr("Enable OpenMP")
                // checked: myApp.enableSpecialCharacter
                checked: (gta_sa.builtWithOpenMP ? true: false)
                enabled: gta_sa.builtWithOpenMP
                onToggled: {
                    if (enableOpenMP.checkState)
                    {
                        nbrThreadValue.value = gta_sa.max_thread_support()
                    } else {
                    nbrThreadValue.value = 1
                }
                // myApp.enableSpecialCharacter = enableSpecialCharacter.checkState
            }
        }
    }
    RowLayout {
        enabled: (gta_sa.builtWithOpenMP ? enableOpenMP.checkState: false)
        Label {
            text: qsTr("CPU core: ")
        }
        Slider {
            id: nbrThreadValue
            value: gta_sa.nbrThreadValue
            stepSize: 1
            from: 1
            to: gta_sa.max_thread_support()
            snapMode: Slider.SnapAlways
        }

        Binding {
            target: gta_sa
            property: "nbrThreadValue"
            value: nbrThreadValue.value
        }
        Label {
            text: (gta_sa.nbrThreadValue
            >= 10) ? gta_sa.nbrThreadValue: "0" + gta_sa.nbrThreadValue
        }
    }
}
}
}
RowLayout {
    Layout.alignment: Qt.AlignHCenter
    GroupBox {
        title: qsTr("Result")
        Layout.alignment: Qt.AlignHCenter
        ColumnLayout {
            /*
            Row {
                Button {
                    text: qsTr("Update List Model")
                    onClicked: myModel.addPerson()
                }
            }*/
            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                TableView {
                    width: 440
                    height: 360
                    columnSpacing: 1
                    rowSpacing: 1
                    clip: true
                    ScrollIndicator.horizontal: ScrollIndicator {}
                    ScrollIndicator.vertical: ScrollIndicator {}
                    model: myModel
                    delegate: Rectangle {
                    implicitWidth: 110
                    implicitHeight: 20
                    border.color: "black"
                    border.width: 2
                    color: heading ? 'antiquewhite': "aliceblue"
                    Text {
                        text: tabledata
                        font.pointSize: 10
                        font.bold: heading ? true: false
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }
}
}

RowLayout {
    Layout.alignment: Qt.AlignHCenter
    GroupBox {
        title: qsTr("Export to JSON")
        Layout.alignment: Qt.AlignHCenter
        ColumnLayout {
            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                Button {
                    Layout.alignment: Qt.AlignHCenter
                    text: qsTr("Export data")
                    enabled: false
                }
            }
        }
    }
    GroupBox {
        title: qsTr("Export to CSV")
        Layout.alignment: Qt.AlignHCenter
        ColumnLayout {
            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                Button {
                    Layout.alignment: Qt.AlignHCenter
                    text: qsTr("Export data")
                    enabled: false
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
