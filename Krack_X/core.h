#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <QString>
//#include <thread>

#ifndef CORE_H
#define CORE_H

class CoreProcessing
{
private:


public:
    CoreProcessing();//Pas de crochet !
    auto GetTime();// { return std::chrono::system_clock::now(); }
    unsigned int CPUThreadCount();
    unsigned int UseCore;
    std::string CurrentFileOpen();
    void exec();
    [[ noreturn ]] void call_from_thread(int tid);

};

#endif // CORE_H
/*QProcess process;
process.start("/path/to/test.sh");
process.waitForFinished();
QString output = process.readAllStandardOutput();
qDebug() << output;
QString err = process.readAllStandardError();
qDebug() << err;
*/
