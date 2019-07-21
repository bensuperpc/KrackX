import QtQuick 2.12
import QtQuick.Controls 2.5

Page {
    id:settings
    width: 600
    height: 400

    title: qsTr("Settings")

    Label {
        text: qsTr("You are on Settings.")
        anchors.centerIn: parent
    }

    Button {
        id: button
        x: 162
        y: 68
        text: qsTr("Button")
    }
}
