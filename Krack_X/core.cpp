#include "core.h"
#include "core_add.h"

#include <iostream>
#include <thread>
#include<qprocess.h>

//#include<chrono>
//#include<random>


#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <QString>
#include <QProcess>
#include <QtDebug>
//QProcess::execute("google-chrome");
static CoreAdd coreadd;
// Date constructor
using namespace std;

CoreProcessing::CoreProcessing()
{
    UseCore = std::thread::hardware_concurrency();//Retourne le nombre de coeur sur la machine et mettre dans UseCore
    OuputConsole = "Test";
}

unsigned int CoreProcessing::CPUThreadCount(){
    return std::thread::hardware_concurrency();//Retourne le nombre de coeur sur la machine
}

std::string CurrentFileOpen(){
    return "";
}

void CoreProcessing::exec(string LauncedP){
    std::thread *tt = new std::thread[UseCore];//On créer les Threads
    for (unsigned int i = 0; i < UseCore; ++i) {
        tt[i] = std::thread(&CoreProcessing::call_from_thread,this,i,LauncedP);
    }
    for (unsigned int i = 0; i < UseCore; ++i){
        tt[i].join();
    }
    delete [] tt;//On nettoie la Memoire

}

[[ noreturn ]] void CoreProcessing::call_from_thread(int tid, string _LauncedP) {

    qDebug() << "Launched by thread " << tid;
    qDebug() << "Program Launch :" << QString::fromStdString(_LauncedP);
    //qDebug() << "Launched by thread " << coreadd.RandomNbrs();outputoutput

    QProcess process;

    //process.start(QString::fromStdString(_LauncedP));
    //process.waitForFinished(-1); // will wait forever until finished

    process.startDetached(QString::fromStdString(_LauncedP));

    QString stdout = process.readAllStandardOutput();
    QString stderr = process.readAllStandardError();

    qDebug() << "#########################";
    qDebug() << stdout;
    qDebug() << "*************************";
    qDebug() << stderr;
    qDebug() << "#########################";
}
