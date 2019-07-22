import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.13

ApplicationWindow {
    property bool isOpened: false
    Material.theme: Material.Dark
    Material.primary: Material.Amber
    id: window

    visible: true
    width: 360
    height: 520
    title: qsTr("Side Panel")

    readonly property bool inPortrait: window.width < window.height

    header: ToolBar {
        RowLayout {
            spacing: 20
            anchors.fill: parent
            ToolButton {
                id: toolButton1
                //text: "\u2630" // symbole représentant le panneau ou "\u2261"
                text: stackView.depth > 1 ? "\u25C0" : "\u2630"
                font.pixelSize: Qt.application.font.pixelSize * 1.6
                onClicked: {
                    console.log("onClicked " + toolButton1.text)
                    if (stackView.depth > 1) {
                        stackView.pop()
                    } else {
                        if (isOpened) {
                            console.log("drawer close()")
                            drawer.close()
                        } else {
                            console.log("drawer open()")
                            drawer.open()
                        }
                    }
                }
            }
            Label {
                text: qsTr("Mon Application")
                horizontalAlignment: Qt.AlignHCenter
                verticalAlignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }
            ToolButton {
                id: toolButton2
                text: "\u22ee" // symbole représentant le menu options ou qsTr("⋮")
                onClicked: menu.open()
            }
        }
    }

    Menu {
        id: menu
        x: parent.width - width
        transformOrigin: Menu.TopRight
        MenuItem {
            id: parametres
            text: "Paramètres"
            onTriggered: {
                console.log("onTriggered " + parametres.text)
                stackView.push("SettingsPage.qml")
            }
        }
        MenuItem {
            id: about
            text: "À propos"
            onTriggered: {
                console.log("onTriggered " + about.text)
                aPropos.open()
            }
        }
    }
    Dialog {
        id: aPropos
        modal: true
        focus: true
        title: "À propos"
        x: (window.width - width) / 2
        y: window.height / 6
        width: Math.min(window.width, window.height) / 3 * 2
        contentHeight: message.height
        Label {
            id: message
            width: aPropos.availableWidth
            text: "Application réalisée en Qt avec le module Quick Controls 2."
            wrapMode: Label.Wrap
            font.pixelSize: 12
        }
    }

    Drawer {
        id: drawer

        width: window.width * 0.66 // occupera 66% de la largeur
        height: window.height // et sur toute la hauteur
        onOpened: {
            console.log("drawer onOpened")
            isOpened = true
        }
        onClosed: {
            console.log("drawer onClosed")
            isOpened = false
        }
        Flickable {
            contentHeight: idContentColumn.height
            anchors.fill: parent
            Column {
                width: parent.width
                height: parent.height
                id: idContentColumn
                ItemDelegate {
                    id: choix1
                    text: qsTr("Settings")
                    width: parent.width // toute la largeur du tiroir
                    onClicked: {
                        console.log("onClicked " + choix1.text)
                        stackView.push("SettingsPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    id: choix2
                    text: qsTr("KrackPasswordPage")
                    width: parent.width
                    onClicked: {
                        console.log("onClicked " + choix2.text)
                        stackView.push("KrackPasswordPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
            }
        }
    }

    Flickable {
        id: flickable

        anchors.fill: parent
        //anchors.topMargin: overlayHeader.height
        anchors.leftMargin: !inPortrait ? drawer.width : undefined

        topMargin: 20
        bottomMargin: 20
        //contentHeight: column.height
        contentHeight: stackView.height
        StackView {
            id: stackView
            initialItem: "mainPage.qml"
            anchors.fill: parent
        }

        //initialItem: qrc:/config_app.qml:6 Expected type name
        ScrollIndicator.vertical: ScrollIndicator {
        }
    }
}
