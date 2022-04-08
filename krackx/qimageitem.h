#ifndef QIMAGEITEM_H
#define QIMAGEITEM_H

#include <QQuickPaintedItem>
#include <QQuickItem>
#include <QObject>
#include <QPainter>
/*

#include <QQuickPaintedItem>
#include <QObject>

class QImageItem : public QQuickPaintedItem
{
    Q_OBJECT
public:
    QImageItem();
};
*/

class QImageItem : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QImage image READ image WRITE setImage NOTIFY imageChanged)

public:
    explicit QImageItem(QQuickItem *parent = Q_NULLPTR) : QQuickPaintedItem(parent) {}

    QImage image() const { return m_image; }
    void setImage(const QImage &image);

    void paint(QPainter *painter) Q_DECL_OVERRIDE;

private:
    QImage m_image;
signals:
    void imageChanged();
};

#endif // QIMAGEITEM_H
