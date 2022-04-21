#include "counter.h"

counter::counter(QObject* parent)
    : QObject {parent}
{
}

void counter::setValue(long long value)
{
  if (value == m_Value)
    return;
  m_Value = value;
  emit valueChanged(value);
}
