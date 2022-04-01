import QtQuick
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Controls.Material

Page {
    title: qsTr("Settings")
    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 5
        GroupBox {
            title: qsTr("Password")
            Column {
                Row {
                    CheckBox {
                        id: enableNumber
                        text: qsTr("123")
                        checked: myApp.enableNumber
                        onToggled: {
                            myApp.enableNumber = enableNumber.checkState
                        }
                    }
                    CheckBox {
                        id: enableSmallAlphabet
                        text: qsTr("abc")
                        checked: myApp.enableSmallAlphabet
                        onToggled: {
                            myApp.enableSmallAlphabet = enableSmallAlphabet.checkState
                        }
                    }
                    CheckBox {
                        id: enableBigAlphabet
                        text: qsTr("ABC")
                        checked: myApp.enableBigAlphabet
                        onToggled: {
                            myApp.enableBigAlphabet = enableBigAlphabet.checkState
                        }
                    }
                }

                CheckBox {
                    id: enableSpecialCharacter
                    text: qsTr("Special character")
                    checked: myApp.enableSpecialCharacter
                    onToggled: {

                        // myApp.enableSpecialCharacter = enableSpecialCharacter.checkState
                    }
                }

                Binding {
                    target: myApp
                    property: "enableSpecialCharacter"
                    value: enableSpecialCharacter.checkState
                }

                CheckBox {
                    id: enableUTF8
                    text: qsTr("UTF-8")
                    checked: myApp.enableUTF8
                    enabled: false
                    onToggled: {
                        myApp.enableUTF8 = enableUTF8.checkState
                    }
                }
            }
        }
        GroupBox {
            title: qsTr("Hardware")
            Column {
                Row {
                    RadioButton {
                        checked: true
                        enabled: true
                        text: qsTr("CPU    ")
                    }
                    RadioButton {
                        enabled: false
                        text: qsTr("CPU/GPU")
                    }
                    RadioButton {
                        enabled: false
                        text: qsTr("GPU    ")
                    }
                }
                Row {
                    Label {
                        text: qsTr("CPU core :")
                    }
                    Slider {
                        id: cpucorecount_slider
                        value: myApp.threadSupport()
                        stepSize: 1.0
                        from: 1
                        to: 64
                        onValueChanged: {
                            cpucorecount_label.text = cpucorecount_slider.value
                            console.log("cpucorecount_slider moved")
                        }
                    }
                    Label {
                        id: cpucorecount_label
                        text: myApp.threadSupport()
                    }
                }
            }
        }
        GroupBox {
            title: qsTr("Hardware")
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
