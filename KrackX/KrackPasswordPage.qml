import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.13
import QtCharts 2.13

Page {
    title: qsTr("KrackPasswordPage")

    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        ComboBox {
            id: comboBox1
            //x: 258
            //y: 54
            model: myApp.comboList
            editable: false
            onAccepted: {
                if (editableCombo.find(currentText) === -1) {
                    model.append({
                                     "text": editText
                                 })
                    currentIndex = editableCombo.find(editText)
                }
            }
        }

        Button {
            id: button1
            text: qsTr("Remove Item")
            onClicked: myApp.removeElement(comboBox1.currentIndex)
        }

        TextField {
            id: textEdit1
            text: qsTr("Text Edit")
            font.pixelSize: 12
        }

        Button {
            id: button2
            text: qsTr("Add Item")
            onClicked: myApp.addElement(textEdit1.text)
        }
        TextField {
            id: textfield1
            placeholderText: qsTr("Enter name 1")
            onTextChanged: {
                myApp.author = textfield1.text
                //textfield2.text = textfield1.text
            }
        }
        TextField {
            id: textfield2
            placeholderText: qsTr("Enter name 2")
            text: myApp.author
        }


        ChartView {
            title: "Line"
            antialiasing: true
            width: 400
            height: 300
            LineSeries {
                name: "LineSeries"
                XYPoint {
                    x: 0
                    y: 0
                }
                XYPoint {
                    x: 1.1
                    y: 2.1
                }
                XYPoint {
                    x: 1.9
                    y: 3.3
                }
                XYPoint {
                    x: 2.1
                    y: 2.1
                }
                XYPoint {
                    x: 2.9
                    y: 4.9
                }
                XYPoint {
                    x: 3.4
                    y: 3.0
                }
                XYPoint {
                    x: 4.1
                    y: 3.3
                }
            }
        }
    }
} //Thank https://stackoverflow.com/questions/26887373/qml-dynamic-combobox-entrys
