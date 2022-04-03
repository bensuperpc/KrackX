//#include <QGuiApplication>
#include <QApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QStringList>
#include <QtCharts>
#include <QtWebEngineQuick>

#include <cmath>
#include <thread>

#include "applicationui.h"

#include "about_compilation.h"

#include "counter.h"

#include "chartdatamodel.h"

void point_generator_proc(MyDataModel *model) {
  for (double t = 0; t < 200; t += 1) {
    double y = (1 + sin(t / 10.0)) / 2.0;

    model->handleNewPoint(QPointF(t, y));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int main(int argc, char *argv[]) {

  QCoreApplication::setOrganizationName("QtExamples");
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  QtWebEngineQuick::initialize();
  // QGuiApplication app(argc, argv);
  // Support chart
  QApplication app(argc, argv);

  QQmlApplicationEngine engine;
  QQmlContext *context = engine.rootContext();

  Applicationui appui;

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

  // Add C++ instance in QML engine
  context->setContextProperty("myApp", &appui);

  auto myDataModel = new MyDataModel();
  auto mapper = new QVXYModelMapper();
  mapper->setModel(myDataModel);
  mapper->setXColumn(0);
  mapper->setYColumn(1);

  std::thread point_generator_thread(point_generator_proc, myDataModel);
  point_generator_thread.detach();

  context->setContextProperty("mapper", mapper);

  about_compilation ac;

  // some more context properties
  // appui.addContextProperty(context);
  context->setContextProperty("myModel",
                              QVariant::fromValue(appui.comboList()));
  // engine.rootContext()->setContextProperty("qtversion", QString(qVersion()));
  context->setContextProperty("qtversion", QString(qVersion()));

  context->setContextProperty("about_compilation", &ac);

  counter myCounter;

  /* Below line makes myCounter object and methods available in QML as
   * "MyCounter" */
  context->setContextProperty("MyCounter", &myCounter);

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
