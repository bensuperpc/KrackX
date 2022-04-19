//#include <QGuiApplication>
#include <QApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QStringList>
#include <QtCharts>
// #include <QtWebEngineQuick>

#include <cmath>
#include <thread>

#include "applicationui.h"

#include "about_compilation.h"

#include "counter.h"

#include "chartdatamodel.h"

#include <QQuickImageProvider>

#include "imageprovider.h"
#include "liveimage.h"

#include "gta_sa_ui.h"

#include "TableModel.h"

void point_generator_proc(MyDataModel *model) {
  for (double t = 0; t < 200; t += 1) {
    double y = (1 + sin(t / 10.0)) / 2.0;

    model->handleNewPoint(QPointF(t, y));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

class ColorImageProvider : public QQuickImageProvider {
public:
  ColorImageProvider() : QQuickImageProvider(QQuickImageProvider::Pixmap) {}

  QPixmap requestPixmap(const QString &id, QSize *size,
                        const QSize &requestedSize) override {
    int width = 100;
    int height = 50;

    if (size)
      *size = QSize(width, height);
    QPixmap pixmap(requestedSize.width() > 0 ? requestedSize.width() : width,
                   requestedSize.height() > 0 ? requestedSize.height()
                                              : height);
    pixmap.fill(QColor(id).rgba());
    return pixmap;
  }
};

int main(int argc, char *argv[]) {
  /*
QCoreApplication::setOrganizationName("QtExamples");
QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
QtWebEngineQuick::initialize();
*/

#if ((QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)) && (QT_VERSION < QT_VERSION_CHECK(6, 0, 0)))
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif

  // QGuiApplication app(argc, argv);
  // Support chart
  QApplication app(argc, argv);

  Applicationui appui;
  about_compilation ac;
  counter myCounter;

  GTA_SA_UI gta_sa_ui;

  // Combobox
  QStringList tmp;
  tmp << "1"
      << "2"
      << "3"
      << "4"
      << "5"
      << "6"
      << "7";
  appui.setComboList(tmp);

  auto myDataModel = new MyDataModel();
  auto mapper = new QVXYModelMapper();
  mapper->setModel(myDataModel);
  mapper->setXColumn(0);
  mapper->setYColumn(1);


  ImageProvider provider{};

  QTimer::singleShot(500, [&provider]() {
      QImage image{480, 480, QImage::Format_ARGB32};
      image.fill(Qt::yellow);
      provider.setImage(std::move(image));
  });

  QTimer::singleShot(1500, [&provider]() {
      std::string str = "/run/media/bensuperpc/MainT7/0u6xvehkj9r71.jpg";
      provider.setImage(str);
  });

  QQmlApplicationEngine engine; // Create Engine AFTER initializing the classes
  QQmlContext *context = engine.rootContext();

  // Add C++ instance in QML engine
  context->setContextProperty("myApp", &appui);

  context->setContextProperty("gta_sa", &gta_sa_ui);

  std::thread point_generator_thread(point_generator_proc, myDataModel);
  point_generator_thread.detach();

  context->setContextProperty("mapper", mapper);

  // some more context properties
  // appui.addContextProperty(context);
  context->setContextProperty("myModel",
                              QVariant::fromValue(appui.comboList()));
  // engine.rootContext()->setContextProperty("qtversion", QString(qVersion()));
  context->setContextProperty("qtversion", QString(qVersion()));

  context->setContextProperty("about_compilation", &ac);

  /* Below line makes myCounter object and methods available in QML as
   * "MyCounter" */
  context->setContextProperty("MyCounter", &myCounter);

  engine.addImageProvider(QLatin1String("colors"), new ColorImageProvider);

  qmlRegisterType<LiveImage>("MyApp.Images", 1, 0, "LiveImage");
  engine.rootContext()->setContextProperty("LiveImageProvider", &provider);


  engine.rootContext()->setContextProperty("myModel", &gta_sa_ui.tableModel);

  const QUrl url(u"qrc:/krackx/main.qml"_qs);
  QObject::connect(
      &engine, &QQmlApplicationEngine::objectCreated, &app,
      [url](const QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
          QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);
  engine.load(url);

  return app.exec();
}
