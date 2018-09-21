#include "core_add.h"
#include<chrono>
#include<random>

CoreAdd::CoreAdd(){}

long CoreAdd::RandomNbrs(){
    typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();

        // obtain a seed from the timer
        myclock::duration d = myclock::now() - beginning;
        unsigned seed2 = d.count();
        std::mt19937 generator (seed2);   // mt19937 is a standard mersenne_twister_engine
    return generator();
}




/*
 *
 *     typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();

    // obtain a seed from the timer
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();
    std::mt19937 generator (seed2);   // mt19937 is a standard mersenne_twister_engine
    qDebug() << "A time seed produced: " << generator();
 */

