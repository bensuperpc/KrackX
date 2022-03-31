#include "applicationui.h"

Applicationui::Applicationui(QObject *parent) : QObject(parent)
{

}
void Applicationui::addContextProperty(QQmlContext *context)
{
    //context->setContextProperty("DataManager", text2);
}

void Applicationui::console(QString st){
    qDebug() << st;
}

const QStringList Applicationui::comboList()
{
    return m_comboList;
}

void Applicationui::setComboList(const QStringList &comboList)
{

    if (m_comboList != comboList)
    {
        m_comboList = comboList;
        emit comboListChanged();
    }

}

int Applicationui::count()
{
    return m_count;
}

void Applicationui::setCount(int cnt)
{
    if (cnt != m_count)
    {
        m_count = cnt;
        emit countChanged();
    }
}

void Applicationui::addElement(const QString &element)
{
    m_comboList.append(element);
    emit comboListChanged();
    setCount(m_comboList.count());
    emit countChanged();

    for (int i = 0; i<m_count; i++)
    {
        qDebug() << m_comboList.at(i);
    }
}

void Applicationui::removeElement(int index)
{
    if (index < m_comboList.count())
    {
        m_comboList.removeAt(index);
        emit comboListChanged();
        setCount(m_comboList.count());
        emit countChanged();
    }

    for (int i = 0; i<m_count; i++)
    {
        qDebug() << m_comboList.at(i);
    }
}
