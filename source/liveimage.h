#ifndef LIVEIMAGE_H
#define LIVEIMAGE_H

#include <QImage>
#include <QPainter>
#include <QQuickPaintedItem>
#include <iostream>

class LiveImage : public QQuickPaintedItem
{
  Q_OBJECT
  Q_PROPERTY(QImage image MEMBER m_image WRITE setImage)
private:
  // Just storage for the image
  QImage m_image;

public:
  explicit LiveImage(QQuickItem* parent = nullptr);
  void setImage(const QImage& image);
  void paint(QPainter* painter) override;
  bool rescale = true;
};

#endif  // SOURCE_LIVEIMAGE_H_
