#ifndef APPLICATIONUI_H
#define APPLICATIONUI_H

#include <QObject>
#include <QQmlContext>
#include <QDebug>

class Applicationui : public QObject
{
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


signals:
    void authorChanged();

private:
    QString m_author;
public slots:
};

#endif // APPLICATIONUI_H
