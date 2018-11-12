#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <QString>
#include <QMainWindow>
#include <QObject>
//#include <thread>
using namespace std;
#ifndef CORE_H
#define CORE_H

class CoreProcessing
{

public:
    QString _CurrentFileOpen;


    CoreProcessing();//Pas de crochet !
    auto GetTime();// { return std::chrono::system_clock::now(); }
    unsigned int CPUThreadCount();
    unsigned int UseCore;
    QString CurrentFileOpen();
    QString OutputConsole;
    void exec(string LauncedP);
    [[ noreturn ]] void call_from_thread(int tid, string _LauncedP);

    bool UTF8_Password;//si le MDP utilise de l'UTF-8
    bool CharSpe_Password;//Si le MDP contient des charactères spéciaux
    bool AttacByDico;//Une attaque par le dictionaire
private:

};





#endif // CORE_H

