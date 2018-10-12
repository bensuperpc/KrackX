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
private:


public:
    CoreProcessing();//Pas de crochet !
    auto GetTime();// { return std::chrono::system_clock::now(); }
    unsigned int CPUThreadCount();
    unsigned int UseCore;
    std::string CurrentFileOpen();
    QString OutputConsole;
    void exec(string LauncedP);
    [[ noreturn ]] void call_from_thread(int tid, string _LauncedP);

    bool UTF8_Password;//si le MDP utilise de l'UTF-8
    bool CharSpe_Password;//Si le MDP contient des charactères spéciaux
    bool AttacByDico;//Une attaque par le dictionaire


};
class StringWrapper : public QObject
{
   Q_OBJECT
public:
    explicit StringWrapper(QObject *parent = nullptr);

    void SetString(const QString& str);

private:
    QString m_str;

signals:
    void TextChanged(QString str);

public slots:
};




#endif // CORE_H

