#include "applicationui.h"

Applicationui::Applicationui(QObject *parent) : QObject(parent)
{

}
void Applicationui::addContextProperty(QQmlContext *context)
{
}

void Applicationui::console(QString st){
    qDebug() << st;
}
