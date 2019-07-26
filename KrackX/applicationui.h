#ifndef APPLICATIONUI_H
#define APPLICATIONUI_H

#include <QObject>
#include <QQmlContext>
#include <QDebug>
#include <QStringList>

#include "core.h"


class Applicationui : public QObject
{
    core coreapp;

    Q_OBJECT

    //For textbox
    Q_PROPERTY(QString author READ author WRITE setAuthor NOTIFY authorChanged)

    //For combobox
    Q_PROPERTY(QStringList comboList READ comboList WRITE setComboList NOTIFY comboListChanged)
    Q_PROPERTY(int count READ count WRITE setCount NOTIFY countChanged)


public:
    explicit Applicationui(QObject *parent = nullptr);

    //For textbox
    void setAuthor(const QString &a) {
        if (a != m_author) {
            m_author = a;
            emit authorChanged();
        }
    }

    //For textbox
    QString author() const {
        return m_author;
    }

    Q_INVOKABLE
    void addContextProperty(QQmlContext* context);

    Q_INVOKABLE
    void console(QString i);

    Q_INVOKABLE
    QString text = "";

    Q_INVOKABLE
    QString text2 = "";

    Q_INVOKABLE
    unsigned threadSupport(){
        return std::thread::hardware_concurrency();
    }

    //For combobox
    const QStringList comboList();
    void setComboList(const QStringList &comboList);

    int count();
    void setCount(int cnt);

    Q_INVOKABLE void addElement(const QString &element);
    Q_INVOKABLE void removeElement(int index);



signals:
    //For textbox
    void authorChanged();

    //For combobox
    void comboListChanged();
    void countChanged();


private:
    //For textbox
    QString m_author;
    //For comobobox
    QStringList m_comboList;
    int         m_count;
public slots:
};

#endif // APPLICATIONUI_H
