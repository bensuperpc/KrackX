#include "scheduler.h"

scheduler::scheduler(QObject *parent) : QObject(parent)
{
    instructions_list.emplace_back();
    //for(unsigned long long i = 0 ; i<=10;i++){}

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
    for(unsigned long long i = 0 ; i<=10;i++){
        //Use ry_unlock later
        g_pages_mutex.lock();
       instructions_list[0].i = instructions_list[0].i + 1;
       qDebug() << "i =" << instructions_list[0].i;
       g_pages_mutex.unlock();
    }

}
