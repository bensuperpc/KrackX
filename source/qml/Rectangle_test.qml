import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

// More info: https: //doc.qt.io/qt-5/qml-qtquick-controls2-scrollbar.html
Page {
    title: qsTr("Rectangle Page")
    Rectangle {
        id: frame
        clip: true
        border.color: "black"
        anchors.fill: parent

        Text {
            id: content
            text: "ABC"
            font.pixelSize: 160
            x: -hbar.position * width
            y: -vbar.position * height
        }

        ScrollBar {
            id: vbar
            hoverEnabled: true
            active: hovered || pressed
            orientation: Qt.Vertical
            size: frame.height / content.height
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.bottom: parent.bottom
        }

        ScrollBar {
            id: hbar
            hoverEnabled: true
            active: hovered || pressed
            orientation: Qt.Horizontal
            size: frame.width / content.width
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
        }
    }
}
