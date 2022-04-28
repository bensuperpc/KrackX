#ifndef _GTA_SA_UI_H_
#define _GTA_SA_UI_H_

#include <QEventLoop>
#include <QObject>
#include <QString>
#include <iostream>
#include <thread>
#include <vector>

#include "TableModel.h"
#include "gta_sa.h"
#include "utils.h"

class GTA_SA_UI : public QObject
{
  Q_OBJECT
  Q_PROPERTY(uint64_t minRangeValue READ minRangeValue WRITE setMinRangeValue NOTIFY minRangeValueChanged)
  Q_PROPERTY(uint64_t maxRangeValue READ maxRangeValue WRITE setMaxRangeValue NOTIFY maxRangeValueChanged)

  Q_PROPERTY(uint64_t nbrThreadValue READ nbrThreadValue WRITE setNbrThreadValue NOTIFY nbrThreadValueChanged)

  Q_PROPERTY(QString buttonValue READ buttonValue WRITE setButtonValue NOTIFY buttonValueChanged)

  Q_PROPERTY(bool use_openmp READ use_openmp WRITE set_use_openmp NOTIFY use_openmp_changed)

  Q_PROPERTY(bool builtWithOpenMP READ builtWithOpenMP CONSTANT)

  // Q_PROPERTY(bool value READ value WRITE enableOpenMPValue NOTIFY
  // enableOpenMPValueChanged) Q_PROPERTY(bool value READ value WRITE
  // enableOpenMPValue NOTIFY enableOpenMPValueChanged)

public:
  explicit GTA_SA_UI(QObject* parent = nullptr);
  GTA_SA gta_sa;
  TableModel tableModel;

  uint64_t minRangeValue() const
  {
    return _minRangeValue;
  };
  uint64_t maxRangeValue() const
  {
    return _maxRangeValue;
  };

  uint64_t nbrThreadValue() const
  {
    return _nbrThreadValue;
  };

  bool use_openmp() const
  {
    return _use_openmp;
  };

  QString buttonValue() const
  {
    return _buttonValue;
  };

  Q_INVOKABLE
  void runOp();
  void runOpThread();

  Q_INVOKABLE
  uint64_t max_thread_support()
  {
    return gta_sa.max_thread_support();
  }

#if defined(_OPENMP)
  bool builtWithOpenMP() const
  {
    return true;
  };
#else
  bool builtWithOpenMP() const
  {
    return false;
  };
#endif

public slots:
  void setMinRangeValue(uint64_t value);
  void setMaxRangeValue(uint64_t value);

  void setNbrThreadValue(uint64_t value);
  void set_use_openmp(bool value);

  void setButtonValue(QString value);

signals:
  void minRangeValueChanged(uint64_t value);
  void maxRangeValueChanged(uint64_t value);

  void nbrThreadValueChanged(uint64_t value);
  void use_openmp_changed(bool value);

  void buttonValueChanged(QString value);

private:
  uint64_t& _minRangeValue = gta_sa.min_range;
  uint64_t& _maxRangeValue = gta_sa.max_range;
  QString _buttonValue = "Launch Bruteforce";
  uint64_t& _nbrThreadValue = gta_sa.num_thread;
  bool& _use_openmp = gta_sa.use_openmp;
};

#endif  // GTA_SA_UI_H
