/****************************************************************************
** Meta object code from reading C++ file 'applicationui.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.13.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../KrackX/applicationui.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'applicationui.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.13.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Applicationui_t {
    QByteArrayData data[18];
    char stringdata0[185];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Applicationui_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Applicationui_t qt_meta_stringdata_Applicationui = {
    {
QT_MOC_LITERAL(0, 0, 13), // "Applicationui"
QT_MOC_LITERAL(1, 14, 13), // "authorChanged"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 16), // "comboListChanged"
QT_MOC_LITERAL(4, 46, 12), // "countChanged"
QT_MOC_LITERAL(5, 59, 18), // "addContextProperty"
QT_MOC_LITERAL(6, 78, 12), // "QQmlContext*"
QT_MOC_LITERAL(7, 91, 7), // "context"
QT_MOC_LITERAL(8, 99, 7), // "console"
QT_MOC_LITERAL(9, 107, 1), // "i"
QT_MOC_LITERAL(10, 109, 13), // "threadSupport"
QT_MOC_LITERAL(11, 123, 10), // "addElement"
QT_MOC_LITERAL(12, 134, 7), // "element"
QT_MOC_LITERAL(13, 142, 13), // "removeElement"
QT_MOC_LITERAL(14, 156, 5), // "index"
QT_MOC_LITERAL(15, 162, 6), // "author"
QT_MOC_LITERAL(16, 169, 9), // "comboList"
QT_MOC_LITERAL(17, 179, 5) // "count"

    },
    "Applicationui\0authorChanged\0\0"
    "comboListChanged\0countChanged\0"
    "addContextProperty\0QQmlContext*\0context\0"
    "console\0i\0threadSupport\0addElement\0"
    "element\0removeElement\0index\0author\0"
    "comboList\0count"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Applicationui[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       3,   70, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x06 /* Public */,
       3,    0,   55,    2, 0x06 /* Public */,
       4,    0,   56,    2, 0x06 /* Public */,

 // methods: name, argc, parameters, tag, flags
       5,    1,   57,    2, 0x02 /* Public */,
       8,    1,   60,    2, 0x02 /* Public */,
      10,    0,   63,    2, 0x02 /* Public */,
      11,    1,   64,    2, 0x02 /* Public */,
      13,    1,   67,    2, 0x02 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // methods: parameters
    QMetaType::Void, 0x80000000 | 6,    7,
    QMetaType::Void, QMetaType::QString,    9,
    QMetaType::UInt,
    QMetaType::Void, QMetaType::QString,   12,
    QMetaType::Void, QMetaType::Int,   14,

 // properties: name, type, flags
      15, QMetaType::QString, 0x00495103,
      16, QMetaType::QStringList, 0x00495103,
      17, QMetaType::Int, 0x00495103,

 // properties: notify_signal_id
       0,
       1,
       2,

       0        // eod
};

void Applicationui::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Applicationui *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->authorChanged(); break;
        case 1: _t->comboListChanged(); break;
        case 2: _t->countChanged(); break;
        case 3: _t->addContextProperty((*reinterpret_cast< QQmlContext*(*)>(_a[1]))); break;
        case 4: _t->console((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: { uint _r = _t->threadSupport();
            if (_a[0]) *reinterpret_cast< uint*>(_a[0]) = std::move(_r); }  break;
        case 6: _t->addElement((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 7: _t->removeElement((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QQmlContext* >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Applicationui::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Applicationui::authorChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Applicationui::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Applicationui::comboListChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Applicationui::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Applicationui::countChanged)) {
                *result = 2;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        auto *_t = static_cast<Applicationui *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QString*>(_v) = _t->author(); break;
        case 1: *reinterpret_cast< QStringList*>(_v) = _t->comboList(); break;
        case 2: *reinterpret_cast< int*>(_v) = _t->count(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        auto *_t = static_cast<Applicationui *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setAuthor(*reinterpret_cast< QString*>(_v)); break;
        case 1: _t->setComboList(*reinterpret_cast< QStringList*>(_v)); break;
        case 2: _t->setCount(*reinterpret_cast< int*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject Applicationui::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_Applicationui.data,
    qt_meta_data_Applicationui,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Applicationui::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Applicationui::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Applicationui.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Applicationui::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 3;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void Applicationui::authorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void Applicationui::comboListChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void Applicationui::countChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
