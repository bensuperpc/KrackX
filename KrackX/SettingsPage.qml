import QtQuick 2.13
import QtQuick.Window 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.13

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
                        text: qsTr("abc")
                        checked: true
                    }
                    CheckBox {
                        text: qsTr("ABC")
                        checked: false
                    }
                }
                CheckBox {
                    text: qsTr("Special characters")
                    checked: false
                }
                CheckBox {
                    text: qsTr("UTF-8")
                    checked: false
                }
            }
        }
        GroupBox {
            title: qsTr("Hardware")
            Column {
                Row {
                    RadioButton {
                        checked: true
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
    }
}
