#ifndef GTA_SA_UI_H
#define GTA_SA_UI_H

#include <QObject>
#include <QString>
#include <QEventLoop>

#include <vector>
#include <iostream>
#include <thread>

#include "gta_sa.h"
#include "TableModel.h"
#include "utils.h"

class GTA_SA_UI : public QObject
{
    Q_OBJECT
    Q_PROPERTY(uint64_t minRangeValue READ minRangeValue WRITE setMinRangeValue NOTIFY minRangeValueChanged)
    Q_PROPERTY(uint64_t maxRangeValue READ maxRangeValue WRITE setMaxRangeValue NOTIFY maxRangeValueChanged)

    Q_PROPERTY(uint64_t nbrThreadValue READ nbrThreadValue WRITE setNbrThreadValue NOTIFY nbrThreadValueChanged)

    Q_PROPERTY(QString buttonValue READ buttonValue WRITE setButtonValue NOTIFY buttonValueChanged)

    Q_PROPERTY(bool builtWithOpenMP READ builtWithOpenMP CONSTANT)

    // Q_PROPERTY(bool value READ value WRITE enableOpenMPValue NOTIFY enableOpenMPValueChanged)
    // Q_PROPERTY(bool value READ value WRITE enableOpenMPValue NOTIFY enableOpenMPValueChanged)

public:
    explicit GTA_SA_UI(QObject *parent = nullptr);
    GTA_SA gta_sa;
    TableModel tableModel;

    uint64_t minRangeValue() const { return _minRangeValue; };
    uint64_t maxRangeValue() const { return _maxRangeValue; };

    uint64_t nbrThreadValue() const { return _nbrThreadValue; };

    QString buttonValue() const { return _buttonValue; };

    Q_INVOKABLE void runOp ();
    void runOpThread();

#if defined(_OPENMP)
    Q_INVOKABLE
    uint64_t threadSupport() { return omp_get_max_threads(); }
#else
    Q_INVOKABLE
        uint64_t threadSupport() { return std::thread::hardware_concurrency(); }
#endif

#if defined(_OPENMP)
    const bool builtWithOpenMP() const { return true; };
#else
        const bool builtWithOpenMP() const { return false; };
#endif

public slots:
    void setMinRangeValue(uint64_t value);
    void setMaxRangeValue(uint64_t value);

    void setNbrThreadValue(uint64_t value);

    void setButtonValue(QString value);

signals:
    void minRangeValueChanged(uint64_t value);
    void maxRangeValueChanged(uint64_t value);

    void nbrThreadValueChanged(uint64_t value);

    void buttonValueChanged(QString value);

private:
    uint64_t _minRangeValue {0} ;
    uint64_t _maxRangeValue {0} ;
    QString _buttonValue = "Launch Bruteforce";
    uint64_t _nbrThreadValue = threadSupport();

};

#endif // GTA_SA_UI_H
