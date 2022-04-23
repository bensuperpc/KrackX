#include "liveimage.h"

LiveImage::LiveImage(QQuickItem* parent)
    : QQuickPaintedItem(parent)
    , m_image {}
{
}

void LiveImage::paint(QPainter* painter)
{
  // Update the image
  const auto size = this->size();
  const auto image_size = this->m_image.size();

  if ((size.width() < image_size.width() || size.height() < image_size.height())
      && (size.width() != 0 || size.height() != 0))
  {
    auto facteur = 1.0;

    auto hauteur = static_cast<float>(image_size.height())
        / static_cast<float>(size.height());
    auto largeur = static_cast<float>(image_size.width())
        / static_cast<float>(size.width());
    facteur = (largeur < hauteur) ? hauteur : largeur;

    painter->drawImage(
        0,
        0,
        this->m_image.scaled((static_cast<int>(image_size.width()) / facteur),
                       (static_cast<int>(image_size.height()) / facteur)));
  } else {
    painter->drawImage(0, 0, m_image);
  }
}

void LiveImage::setImage(const QImage& image)
{
  m_image = image;

  std::cout << "Image updated" << std::endl;
  // Redraw the image
  update();
}
