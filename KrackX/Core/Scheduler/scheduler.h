#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <QObject>

#include <QtDebug>
#include <iostream>

#include <mutex>
#include <thread>

#include "instructions.h"




class scheduler : public QObject
{
    Q_OBJECT
public:
    explicit scheduler(QObject *parent = nullptr);

    void cpu(int threadid);

    Q_INVOKABLE
    unsigned maxThreadSupport(){
        return std::thread::hardware_concurrency();
    }
    std::vector<instructions>  instructions_list;

    std::thread *threads = new std::thread[maxThreadSupport()];

    std::mutex g_pages_mutex;
signals:

public slots:
};

#endif // SCHEDULER_H
