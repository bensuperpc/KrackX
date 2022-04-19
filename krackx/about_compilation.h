#ifndef ABOUT_COMPILATION_H
#define ABOUT_COMPILATION_H

#include <QDateTime>
#include <QObject>
#include <QString>

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class about_compilation : public QObject {
  Q_OBJECT
public:
  explicit about_compilation(QObject *parent = nullptr);

  std::string ver_string(int a, int b, int c);

  std::string true_cxx =
#ifdef __clang__
      "clang++";
#else
      "g++";
#endif

  std::string true_cxx_ver =
#ifdef __clang__
      ver_string(__clang_major__, __clang_minor__, __clang_patchlevel__);
#else
      ver_string(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif

  Q_INVOKABLE
  QString return_Compiler_version() {
    return QString::fromStdString(true_cxx_ver);
  }

  Q_INVOKABLE
  QString return_Compiler_name() { return QString::fromStdString(true_cxx); }

  Q_INVOKABLE
  QString return_Cplusplus_used() { return QString::number(__cplusplus); }

  Q_INVOKABLE
  QString return_BuildDate() { return QString::fromStdString(__DATE__); }
  Q_INVOKABLE
  QString return_BuildTime() { return QString::fromStdString(__TIME__); }
  Q_INVOKABLE
  QString openmpIsEnable() {
#if !defined(_OPENMP)
    return QString::fromStdString("false");
#else
    return QString::fromStdString("true");
#endif
  }
  Q_INVOKABLE
  QString cudaIsEnable() { return QString::fromStdString("false"); }
  Q_INVOKABLE
  QString openclIsEnable() { return QString::fromStdString("false"); }

signals:

public slots:
};

#endif // ABOUT_COMPILATION_H
