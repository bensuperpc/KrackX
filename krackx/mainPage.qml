import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

Page {
    title: qsTr("Main Page")

    /*
    Label {
        text: qsTr("You are on Main Page")
        anchors.centerIn: parent
    }
    */
    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 20

        Text {
            id: labelCount
            color: "white"
            // C++ method Counter::value(). Bound via Q_PROPERTY, updates automatically on change
            text: "Counter: " + MyCounter.value + "."
        }

        Text {
            property int changeCount: 0
            id: labelChanged
            color: "white"
            text: "Count has changed " + changeCount + " times."
            // Receive the valueChanged NOTIFY
            Connections {
                target: MyCounter
                onValueChanged: {
                    ++labelChanged.changeCount
                }
            }
        }

        Row {
            spacing: 20
            Button {
                text: qsTr("Increase Counter")
                onClicked: ++MyCounter.value
            }

            Button {
                text: qsTr("Set counter to 10")
                // C++ method Counter::setValue(long long), bound via Q_PROPERTY
                onClicked: MyCounter.setValue(10)
            }

            Button {
                text: qsTr("Reset")
                onClicked: {
                    // C++ method Counter::setValue(long long), bound via Q_PROPERTY
                    MyCounter.setValue(0)
                }
            }
        }

        Row {
            spacing: 20

            Text {
                id: setText
                color: "white"
                text: qsTr("Enter counter value: ")
            }

            TextField {
                id: counterInput
                focus: true
                placeholderText: qsTr("Enter value")
                text: MyCounter.value
            }

            // Bi-directional binding, entering a number in the textarea updates the
            // C++ class, if the C++ class is updated, the textarea is updated as well.
            Binding {
                target: MyCounter
                property: "value"
                value: counterInput.text
            }
        }
    }
}