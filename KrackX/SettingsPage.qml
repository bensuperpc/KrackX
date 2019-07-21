import QtQuick 2.13
import QtQuick.Window 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.13

Page {
    Column {
        anchors.horizontalCenter: parent.horizontalCenter
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
    }
}
