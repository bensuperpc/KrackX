#ifndef CPU_H
#define CPU_H

#include <iostream>
#include <thread>
#include<qprocess.h>
#include<cpuid.h>

class cpuinfo
{
public:
    cpuinfo();
    unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
};

#endif // CPU_H
