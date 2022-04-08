#include "qimageitem.h"

void QImageItem::setImage(const QImage &image)
{
    m_image = image;
    emit imageChanged();
    update();

    setImplicitWidth(m_image.width());
    setImplicitHeight(m_image.height());
}

void QImageItem::paint(QPainter *painter)
{
    if (m_image.isNull()) return;

    //painter->drawImage(m_image.scaled(width(), height()));
    painter->drawImage(QPointF(0.0f,0.0f), m_image);
}
