#ifndef CORE_H
#define CORE_H

#include <QObject>

#include <iostream>
#include <thread>

#include<qprocess.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <QString>
#include <QProcess>
#include <QtDebug>
#include <iostream>
using namespace std;

class core : public QObject
{
    Q_OBJECT
public:
    explicit core(QObject *parent = nullptr);


    unsigned int CPUThreadCount();
    unsigned int UseCore;
    QString CurrentFileOpen();
    QString OutputConsole;
    void exec(string LauncedP);
    [[ noreturn ]] void call_from_thread(int tid, string _LauncedP);

    bool UTF8_Password;//si le MDP utilise de l'UTF-8
    bool CharSpe_Password;//Si le MDP contient des charactères spéciaux
    bool AttacByDico;//Une attaque par le dictionaire

signals:

public slots:
};

#endif // CORE_H
