//#include <QGuiApplication>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "applicationui.h"

#include <QStringList>

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    //for theme
    qputenv("QT_QUICK_CONTROLS_STYLE", "material");

    //If you use QML app (without QtWidgets)
    //QGuiApplication app(argc, argv);

    //If you use QML and QtWidgets app
    QApplication app(argc, argv);
    //QCoreApplication::setAttribute(Qt::AA_UseSoftwareOpenGL);
    //QCoreApplication::setAttribute(Qt::AA_UseOpenGLES);

    QQmlApplicationEngine engine;

    Applicationui appui;

    //For combobox
    QStringList tmp;
    tmp << "1" << "2" << "3" << "4" << "5" << "6" << "7";
    appui.setComboList(tmp);

    QQmlContext *ownContext = engine.rootContext();
    ownContext->setContextProperty("myModel", QVariant::fromValue(appui.comboList()));


    //Add C++ instance in QML engine
    QQmlContext* context = engine.rootContext();
    context->setContextProperty("myApp", &appui);


    const QUrl url(QStringLiteral("qrc:/main.qml"));

    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
                         if (!obj && url == objUrl)
                             QCoreApplication::exit(-1);
                     }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
