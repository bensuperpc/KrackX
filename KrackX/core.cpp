#include "core.h"
using namespace std;

core::core(QObject *parent) : QObject(parent)
{

}

void core::exec(string LauncedP){
    std::thread *tt = new std::thread[UseCore];//On créer les Threads
    for (unsigned int i = 0; i < UseCore; ++i) {
        tt[i] = std::thread(&core::call_from_thread,this,i,LauncedP);
    }
    for (unsigned int i = 0; i < UseCore; ++i){
        tt[i].join();
    }
    delete [] tt;//On nettoie la Memoire
}

[[ noreturn ]] void core::call_from_thread(int tid, string _LauncedP) {

    qDebug() << "Launched by thread " << tid;
    qDebug() << "Program Launch :" << QString::fromStdString(_LauncedP);
    //qDebug() << "Launched by thread " << coreadd.RandomNbrs();outputoutput

    QProcess process;

    //process.start(QString::fromStdString(_LauncedP));
    //process.waitForFinished(-1); // will wait forever until finished

    process.startDetached(QString::fromStdString(_LauncedP));

    //QString stdout = process.readAllStandardOutput();
    //QString stderr = process.readAllStandardError();

    qDebug() << "#########################";
    qDebug() << stdout;
    qDebug() << "*************************";
    qDebug() << stderr;
    qDebug() << "#########################";
}
