//#include <QGuiApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <QApplication>

#include <QStringList>

#include "applicationui.h"

#include "about_compilation.h"

#include "counter.h"

int main(int argc, char *argv[]) {
  // QGuiApplication app(argc, argv);
  QApplication app(argc, argv);

  QQmlApplicationEngine engine;
  QQmlContext *context = engine.rootContext();

  Applicationui appui;

  // For combobox
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
      [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
          QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);
  engine.load(url);

  return app.exec();
}
