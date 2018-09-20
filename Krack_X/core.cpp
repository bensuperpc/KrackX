#include "core.h"
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
//QProcess::execute("google-chrome");


// Date constructor
CoreProcessing::CoreProcessing()
{
    UseCore = std::thread::hardware_concurrency();


}

unsigned int CoreProcessing::CPUThreadCount(){
    return std::thread::hardware_concurrency();
}

std::string CurrentFileOpen(){
    return "";
}

void CoreProcessing::exec(){
    std::thread *tt = new std::thread[UseCore - 1];//On créer les Threads
    for (unsigned int i = 0; i < UseCore - 1; ++i) {
        tt[i] = std::thread(&CoreProcessing::call_from_thread,this,i);
    }
    for (unsigned int i = 0; i < UseCore - 1; ++i){
        tt[i].join();
    }
    delete [] tt;//On nettoie la Memoire

}
    [[ noreturn ]] void CoreProcessing::call_from_thread(int tid) {
        qDebug() << "Launched by thread " << tid;
        //for(;;){
        //put the code you want to loop forever here.
       // }

}




/*
QProcess process;
process.start("google-chrome");
process.waitForFinished(-1); // will wait forever until finished

QString stdout = process.readAllStandardOutput();
QString stderr = process.readAllStandardError();
qDebug() << stdout;
qDebug() << "*************************";
qDebug() << stderr;
*/

/*
QString exec(const char* cmd) {//https://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix

    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return "";
}
*/
