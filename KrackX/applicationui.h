#ifndef APPLICATIONUI_H
#define APPLICATIONUI_H

#include <QObject>
#include <QQmlContext>
#include <QDebug>

#include "cpuinfo.h"


class Applicationui : public QObject
{
    cpuinfo processorInfo;

    Q_OBJECT
    Q_PROPERTY(QString author READ author WRITE setAuthor NOTIFY authorChanged)

public:
    explicit Applicationui(QObject *parent = nullptr);

    void setAuthor(const QString &a) {
        if (a != m_author) {
            m_author = a;
            emit authorChanged();
        }
    }

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
    unsigned threadSupport(){
        return processorInfo.concurentThreadsSupported;
    }



signals:
    void authorChanged();

private:
    QString m_author;
public slots:
};

#endif // APPLICATIONUI_H
