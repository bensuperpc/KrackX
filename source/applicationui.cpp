#include "applicationui.h"

Applicationui::Applicationui(QObject* parent)
    : QObject(parent)
{
}

void Applicationui::addContextProperty(QQmlContext* context)
{
  // context->setContextProperty("DataManager", text2);
}

// For settings
void Applicationui::setEnableNumber(bool newValue)
{
  if (m_enableNumber != newValue) {
    m_enableNumber = newValue;
    emit enableNumberChanged(newValue);
  }
}

void Applicationui::setEnableSmallAlphabet(bool newValue)
{
  if (m_enableSmallAlphabet != newValue) {
    m_enableSmallAlphabet = newValue;
    emit enableSmallAlphabetChanged(newValue);
  }
}

void Applicationui::setEnableBigAlphabet(bool newValue)
{
  if (m_enableBigAlphabet != newValue) {
    m_enableBigAlphabet = newValue;
    emit enableBigAlphabetChanged(newValue);
  }
}

void Applicationui::setEnableSpecialCharacter(bool newValue)
{
  if (m_enableSpecialCharacter != newValue) {
    m_enableSpecialCharacter = newValue;
    emit enableSpecialCharacterChanged(newValue);
  }
}

void Applicationui::setEnableUTF8(bool newValue)
{
  if (m_enableUTF8 != newValue) {
    m_enableUTF8 = newValue;
    emit enableUTF8Changed(newValue);
  }
}

void Applicationui::quitSignalInvokable()
{
  std::cout << "quitSignalInvokable" << std::endl;
}

void Applicationui::quitSignalSlot()
{
  std::cout << "quitSignalSlot" << std::endl;
}

void Applicationui::console(const QString st)
{
  qDebug() << st;
}

const QStringList Applicationui::comboList()
{
  return m_comboList;
}

void Applicationui::setComboList(const QStringList& comboList)
{
  if (m_comboList != comboList) {
    m_comboList = comboList;
    emit comboListChanged();
  }
}

int Applicationui::count()
{
  return m_count;
}

void Applicationui::setCount(int cnt)
{
  if (cnt != m_count) {
    m_count = cnt;
    emit countChanged();
  }
}

void Applicationui::addElement(const QString& element)
{
  m_comboList.append(element);
  emit comboListChanged();
  setCount(static_cast<int>(m_comboList.count()));
  emit countChanged();

  for (int i = 0; i < m_count; i++) {
    qDebug() << m_comboList.at(i);
  }
}

void Applicationui::removeElement(int index)
{
  if (index < m_comboList.count()) {
    m_comboList.removeAt(index);
    emit comboListChanged();
    setCount(static_cast<int>(m_comboList.count()));
    emit countChanged();
  }

  for (int i = 0; i < m_count; i++) {
    qDebug() << m_comboList.at(i);
  }
}
