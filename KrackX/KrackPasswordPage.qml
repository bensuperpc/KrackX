import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtQuick.Controls.Material 2.13
import QtQml.Models 2.3

Page {
    title: qsTr("KrackPasswordPage")


    /*
    Label {
        text: qsTr("KrackPasswordPage")
        anchors.centerIn: parent
    }
    */
    DelegateModel {

        property var filterAccepts: function(item) {
            return true
        }

        onFilterAcceptsChanged: refilter()

        function refilter() {
            if(hidden.count>0)
                hidden.setGroups(0, hidden.count, "default")
            if(items.count>0)
                items.setGroups(0, items.count, "default")
        }

        function filter() {
            while (unsortedItems.count > 0) {
                var item = unsortedItems.get(0)
                if(filterAccepts(item.model))
                    item.groups = "items"
                else
                    item.groups = "hidden"
            }
        }

        items.includeByDefault: false
        groups: [
            DelegateModelGroup {
                id: default
                name: "default"
                includeByDefault: true
                onChanged: filter()
            },
            DelegateModelGroup {
                id: hidden
                name: "hidden"
            }
        ]

    }
}
