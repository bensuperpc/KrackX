import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Controls.Material 2.13
import QtQuick.Layouts 1.13
import QtQuick.Window 2.13

ApplicationWindow {
    property bool isOpened: false

    //! [orientation]
    property bool inPortrait: window.width < window.height
    //! [orientation]

    //property int y: Screen.desktopAvailableHeight - height
    //property int x: Screen.desktopAvailableWidth - width
    Material.theme: Material.Dark
    Material.primary: Material.Amber
    id: window

    visible: true
    width: 480
    height: 720

    title: qsTr("Side Panel")

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            ToolButton {
                id: toolButton1
                //text: qsTr("\u2630")
                Image {
                    id: iconmenu
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.verticalCenter: parent.verticalCenter
                    height: 24
                    width: 24
                    source: "qrc:images/icons/menu@4x.png"
                }
                onClicked: {
                    if (isOpened) {
                        console.log("drawer close()")
                        drawer.close()
                    } else {
                        console.log("drawer open()")
                        drawer.open()
                    }
                }
            }
            Label {
                id: title_label
                text: stackView.currentItem.title
                elide: Label.ElideRight
                horizontalAlignment: Qt.AlignHCenter
                verticalAlignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }
            ToolButton {
                text: qsTr("⋮")
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

        width: window.width * 0.60 // occupera 66% de la largeur
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
            clip: true
            Column {
                width: parent.width
                height: parent.height
                id: idContentColumn

                ItemDelegate {
                    id: choix1
                    text: qsTr("MainPage")
                    width: parent.width // toute la largeur du tiroir
                    onClicked: {
                        console.log("onClicked " + choix1.text)
                        stackView.push("mainPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    width: parent.width
                    height: menu_separator.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    enabled: false
                    MenuSeparator {
                        id: menu_separator
                        width: parent.width
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }

                ItemDelegate {
                    id: choix2
                    text: qsTr("Settings")
                    width: parent.width // toute la largeur du tiroir
                    onClicked: {
                        console.log("onClicked " + choix1.text)
                        stackView.push("SettingsPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }

                ItemDelegate {
                    id: choix3
                    text: qsTr("KrackPasswordPage")
                    width: parent.width
                    onClicked: {
                        console.log("onClicked " + choix3.text)
                        stackView.push("KrackPasswordPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    width: parent.width
                    height: menu_separator1.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    enabled: false
                    MenuSeparator {
                        id: menu_separator1
                        width: parent.width
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }
                ItemDelegate {
                    id: choix4
                    text: qsTr("About")
                    width: parent.width
                    onClicked: {
                        console.log("onClicked " + choix4.text)
                        stackView.push("AboutPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
            }
        }
    }
    Flickable {
        id: flickable

        anchors.fill: parent
        focus: true
        topMargin: 20
        bottomMargin: 20
        contentHeight: stackView.height
        clip: true
        //boundsBehavior: Flickable.StopAtBounds
        StackView {
            id: stackView
            initialItem: "mainPage.qml"
            anchors.fill: parent
            anchors.centerIn: parent
        }
        ScrollIndicator.vertical: ScrollIndicator {
        }
        //initialItem: qrc:/config_app.qml:6 Expected type name
    }
}

/*
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
    }
}
*/

