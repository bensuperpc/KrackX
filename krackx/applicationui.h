#ifndef APPLICATIONUI_H
#define APPLICATIONUI_H

#include <QDebug>
#include <QObject>
#include <QQmlContext>
#include <QStringList>
#include <thread>
#include <iostream>

class Applicationui : public QObject {
  Q_OBJECT

  // For Settings
  Q_PROPERTY(bool enableNumber READ enableNumber WRITE setEnableNumber NOTIFY
                 enableNumberChanged)
  Q_PROPERTY(bool enableSmallAlphabet READ enableSmallAlphabet WRITE
                 setEnableSmallAlphabet NOTIFY enableSmallAlphabetChanged)
  Q_PROPERTY(bool enableBigAlphabet READ enableBigAlphabet WRITE
                 setEnableBigAlphabet NOTIFY enableBigAlphabetChanged)
  Q_PROPERTY(bool enableSpecialCharacter READ enableSpecialCharacter WRITE
                 setEnableSpecialCharacter NOTIFY enableSpecialCharacterChanged)
  Q_PROPERTY(bool enableUTF8 READ enableUTF8 WRITE setEnableUTF8 NOTIFY
                 enableUTF8Changed)

  // For textbox
  Q_PROPERTY(QString author READ author WRITE setAuthor NOTIFY authorChanged)

  // For combobox
  Q_PROPERTY(QStringList comboList READ comboList WRITE setComboList NOTIFY
                 comboListChanged)
  Q_PROPERTY(int count READ count WRITE setCount NOTIFY countChanged)

public:
  explicit Applicationui(QObject *parent = nullptr);

  // For Settings
  Q_INVOKABLE bool enableNumber() const { return m_enableNumber; };
  Q_INVOKABLE bool enableSmallAlphabet() const {
    return m_enableSmallAlphabet;
  };
  Q_INVOKABLE bool enableBigAlphabet() const { return m_enableBigAlphabet; };
  Q_INVOKABLE bool enableSpecialCharacter() const {
    return m_enableSpecialCharacter;
  };
  Q_INVOKABLE bool enableUTF8() const { return m_enableUTF8; };


  Q_INVOKABLE void quitSignalInvokable();
  Q_SLOT void quitSignalSlot();

  // For textbox
  void setAuthor(const QString &a) {
    if (a != m_author) {
      m_author = a;
      emit authorChanged();
    }
  }

  // For textbox
  QString author() const { return m_author; }

  Q_INVOKABLE
  void addContextProperty(QQmlContext *context);

  Q_INVOKABLE
  void console(const QString);

  Q_INVOKABLE
  QString text = "";

  Q_INVOKABLE
  QString text2 = "";

  Q_INVOKABLE
  unsigned threadSupport() { return std::thread::hardware_concurrency(); }

  // For combobox
  const QStringList comboList();
  void setComboList(const QStringList &comboList);

  int count();
  void setCount(int cnt);

  Q_INVOKABLE void addElement(const QString &element);
  Q_INVOKABLE void removeElement(int index);

private:
  // For textbox
  QString m_author;
  // For comobobox
  QStringList m_comboList;
  int m_count = 0;

  // For settings
  bool m_enableNumber = true;
  bool m_enableSmallAlphabet = true;
  bool m_enableBigAlphabet = true;
  bool m_enableSpecialCharacter = true;
  bool m_enableUTF8 = false;

signals:
  // For Settings
  void enableNumberChanged(bool newValue);
  void enableSmallAlphabetChanged(bool newValue);
  void enableBigAlphabetChanged(bool newValue);
  void enableSpecialCharacterChanged(bool newValue);
  void enableUTF8Changed(bool newValue);

  // For textbox
  void authorChanged();

  // For combobox
  void comboListChanged();
  void countChanged();

public slots:
  // For Settings
  void setEnableNumber(bool value);
  void setEnableSmallAlphabet(bool value);
  void setEnableBigAlphabet(bool value);
  void setEnableSpecialCharacter(bool value);
  void setEnableUTF8(bool value);
};

#endif // APPLICATIONUI_H
