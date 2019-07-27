#include "scheduler.h"

scheduler::scheduler(QObject *parent) : QObject(parent)
{
    for(unsigned long long i = 0 ; i<=10;i++){
        instructions_list.emplace_back();
    }

    for (unsigned int i = 0; i < maxThreadSupport(); ++i) {
        threads[i] = std::thread(&scheduler::cpu,this,i);
    }
    for (unsigned int i = 0; i < maxThreadSupport(); ++i){
        threads[i].join();
    }
    delete[] threads;//On nettoie la Memoire
}

void scheduler::cpu(int threadid){
    qDebug() << "thread id: " << threadid;
}
