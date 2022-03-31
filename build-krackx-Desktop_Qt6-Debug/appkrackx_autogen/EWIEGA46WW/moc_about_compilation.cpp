/****************************************************************************
** Meta object code from reading C++ file 'about_compilation.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.2.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../krackx/about_compilation.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'about_compilation.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.2.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_about_compilation_t {
    const uint offsetsAndSize[14];
    char stringdata0[120];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(offsetof(qt_meta_stringdata_about_compilation_t, stringdata0) + ofs), len 
static const qt_meta_stringdata_about_compilation_t qt_meta_stringdata_about_compilation = {
    {
QT_MOC_LITERAL(0, 17), // "about_compilation"
QT_MOC_LITERAL(18, 23), // "return_Compiler_version"
QT_MOC_LITERAL(42, 0), // ""
QT_MOC_LITERAL(43, 20), // "return_Compiler_name"
QT_MOC_LITERAL(64, 21), // "return_Cplusplus_used"
QT_MOC_LITERAL(86, 16), // "return_BuildDate"
QT_MOC_LITERAL(103, 16) // "return_BuildTime"

    },
    "about_compilation\0return_Compiler_version\0"
    "\0return_Compiler_name\0return_Cplusplus_used\0"
    "return_BuildDate\0return_BuildTime"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_about_compilation[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // methods: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   44,    2, 0x02,    1 /* Public */,
       3,    0,   45,    2, 0x02,    2 /* Public */,
       4,    0,   46,    2, 0x02,    3 /* Public */,
       5,    0,   47,    2, 0x02,    4 /* Public */,
       6,    0,   48,    2, 0x02,    5 /* Public */,

 // methods: parameters
    QMetaType::QString,
    QMetaType::QString,
    QMetaType::QString,
    QMetaType::QString,
    QMetaType::QString,

       0        // eod
};

void about_compilation::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<about_compilation *>(_o);
        (void)_t;
        switch (_id) {
        case 0: { QString _r = _t->return_Compiler_version();
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        case 1: { QString _r = _t->return_Compiler_name();
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        case 2: { QString _r = _t->return_Cplusplus_used();
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        case 3: { QString _r = _t->return_BuildDate();
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        case 4: { QString _r = _t->return_BuildTime();
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    }
}

const QMetaObject about_compilation::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_about_compilation.offsetsAndSize,
    qt_meta_data_about_compilation,
    qt_static_metacall,
    nullptr,
qt_incomplete_metaTypeArray<qt_meta_stringdata_about_compilation_t
, QtPrivate::TypeAndForceComplete<about_compilation, std::true_type>

, QtPrivate::TypeAndForceComplete<QString, std::false_type>, QtPrivate::TypeAndForceComplete<QString, std::false_type>, QtPrivate::TypeAndForceComplete<QString, std::false_type>, QtPrivate::TypeAndForceComplete<QString, std::false_type>, QtPrivate::TypeAndForceComplete<QString, std::false_type>

>,
    nullptr
} };


const QMetaObject *about_compilation::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *about_compilation::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_about_compilation.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int about_compilation::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 5;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
