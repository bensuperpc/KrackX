import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtQuick.Controls.Material 2.13

Page {
    title: qsTr("KrackPasswordPage")
    Column {
        anchors.horizontalCenter: parent.horizontalCenter

        RowLayout {
            Button {
                text: "Ok"
                //onClicked: model.submit()
                onClicked: {

                }
            }
            Button {
                text: "Cancel"
                //onClicked: model.revert()
            }
        }
    }
}
