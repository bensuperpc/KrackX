import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtCharts

Page {
    title: qsTr("KrackPasswordPage")
    property int timeStep: 0

    Column {
        anchors.horizontalCenter: parent.horizontalCenter

        ComboBox {
            id: comboBox1
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
            id: chartView
            antialiasing: true
            width: Window.width * 0.8
            height: Window.height * 0.6

            ValueAxis {
                id: axisX
                min: 0
                max: 400
            }

            Component.onCompleted: {
                mapper.series = series2
            }

            LineSeries {
                id: series1
                axisX: axisX
                name: "From QML"
            }

            LineSeries {
                id: series2
                axisX: axisX
                name: "From C++"
            }
        }

        Timer {
            interval: 100
            repeat: true
            running: true
            onTriggered: {
                timeStep++
                var y = (1 + Math.cos(timeStep / 10.0)) / 2.0
                series1.append(timeStep, y)
            }
        }
    }
} //Thank https://stackoverflow.com/questions/26887373/qml-dynamic-combobox-entrys
