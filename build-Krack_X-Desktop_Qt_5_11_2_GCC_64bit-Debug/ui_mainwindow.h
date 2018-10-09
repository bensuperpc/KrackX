/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionDivers;
    QAction *actionTest;
    QAction *actionDivers_2;
    QAction *actionBug;
    QAction *actionAuteurs;
    QAction *actionOuvrir_un_fichier;
    QAction *actionQuitter;
    QAction *actionEnregister_les_Log;
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QPushButton *pushButton;
    QGroupBox *groupBox_2;
    QSlider *CoreUse_horizontalSlider;
    QLabel *CoreUse_Label;
    QCheckBox *UTF8_CheckBox;
    QCheckBox *CharSpe_CheckBox;
    QCheckBox *AttaqueDico_CheckBox;
    QGroupBox *groupBox_3;
    QTextEdit *textEdit;
    QLineEdit *lineEdit;
    QProgressBar *progressBar;
    QLabel *label;
    QMenuBar *menuBar;
    QMenu *menuFichier;
    QMenu *menuAide;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(630, 323);
        actionDivers = new QAction(MainWindow);
        actionDivers->setObjectName(QStringLiteral("actionDivers"));
        actionTest = new QAction(MainWindow);
        actionTest->setObjectName(QStringLiteral("actionTest"));
        actionDivers_2 = new QAction(MainWindow);
        actionDivers_2->setObjectName(QStringLiteral("actionDivers_2"));
        actionBug = new QAction(MainWindow);
        actionBug->setObjectName(QStringLiteral("actionBug"));
        actionAuteurs = new QAction(MainWindow);
        actionAuteurs->setObjectName(QStringLiteral("actionAuteurs"));
        actionOuvrir_un_fichier = new QAction(MainWindow);
        actionOuvrir_un_fichier->setObjectName(QStringLiteral("actionOuvrir_un_fichier"));
        actionQuitter = new QAction(MainWindow);
        actionQuitter->setObjectName(QStringLiteral("actionQuitter"));
        actionEnregister_les_Log = new QAction(MainWindow);
        actionEnregister_les_Log->setObjectName(QStringLiteral("actionEnregister_les_Log"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 141, 271));
        pushButton = new QPushButton(groupBox);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(10, 30, 121, 23));
        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(160, 10, 141, 271));
        CoreUse_horizontalSlider = new QSlider(groupBox_2);
        CoreUse_horizontalSlider->setObjectName(QStringLiteral("CoreUse_horizontalSlider"));
        CoreUse_horizontalSlider->setGeometry(QRect(10, 50, 121, 16));
        CoreUse_horizontalSlider->setMouseTracking(false);
        CoreUse_horizontalSlider->setMinimum(1);
        CoreUse_horizontalSlider->setMaximum(64);
        CoreUse_horizontalSlider->setPageStep(8);
        CoreUse_horizontalSlider->setOrientation(Qt::Horizontal);
        CoreUse_Label = new QLabel(groupBox_2);
        CoreUse_Label->setObjectName(QStringLiteral("CoreUse_Label"));
        CoreUse_Label->setGeometry(QRect(10, 30, 121, 16));
        UTF8_CheckBox = new QCheckBox(groupBox_2);
        UTF8_CheckBox->setObjectName(QStringLiteral("UTF8_CheckBox"));
        UTF8_CheckBox->setGeometry(QRect(10, 70, 121, 21));
        CharSpe_CheckBox = new QCheckBox(groupBox_2);
        CharSpe_CheckBox->setObjectName(QStringLiteral("CharSpe_CheckBox"));
        CharSpe_CheckBox->setGeometry(QRect(10, 90, 121, 21));
        AttaqueDico_CheckBox = new QCheckBox(groupBox_2);
        AttaqueDico_CheckBox->setObjectName(QStringLiteral("AttaqueDico_CheckBox"));
        AttaqueDico_CheckBox->setGeometry(QRect(10, 110, 121, 21));
        groupBox_3 = new QGroupBox(centralWidget);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(320, 10, 291, 271));
        textEdit = new QTextEdit(groupBox_3);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        textEdit->setGeometry(QRect(10, 30, 271, 171));
        textEdit->setAcceptDrops(false);
        textEdit->setReadOnly(true);
        lineEdit = new QLineEdit(groupBox_3);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(50, 240, 231, 23));
        lineEdit->setReadOnly(true);
        progressBar = new QProgressBar(groupBox_3);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(10, 210, 271, 23));
        progressBar->setValue(0);
        label = new QLabel(groupBox_3);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 240, 41, 21));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 630, 20));
        menuFichier = new QMenu(menuBar);
        menuFichier->setObjectName(QStringLiteral("menuFichier"));
        menuAide = new QMenu(menuBar);
        menuAide->setObjectName(QStringLiteral("menuAide"));
        MainWindow->setMenuBar(menuBar);

        menuBar->addAction(menuFichier->menuAction());
        menuBar->addAction(menuAide->menuAction());
        menuFichier->addAction(actionOuvrir_un_fichier);
        menuFichier->addSeparator();
        menuFichier->addAction(actionEnregister_les_Log);
        menuFichier->addAction(actionQuitter);
        menuAide->addAction(actionDivers_2);
        menuAide->addAction(actionBug);
        menuAide->addAction(actionAuteurs);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        actionDivers->setText(QApplication::translate("MainWindow", "Divers", nullptr));
        actionTest->setText(QApplication::translate("MainWindow", "Test", nullptr));
        actionDivers_2->setText(QApplication::translate("MainWindow", "Divers", nullptr));
        actionBug->setText(QApplication::translate("MainWindow", "Signaler un bug...", nullptr));
        actionAuteurs->setText(QApplication::translate("MainWindow", "A propos", nullptr));
        actionOuvrir_un_fichier->setText(QApplication::translate("MainWindow", "Ouvrir un fichier", nullptr));
        actionQuitter->setText(QApplication::translate("MainWindow", "Quitter", nullptr));
        actionEnregister_les_Log->setText(QApplication::translate("MainWindow", "Enregister les Log", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "Principal", nullptr));
        pushButton->setText(QApplication::translate("MainWindow", "Lancer", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Options", nullptr));
        CoreUse_Label->setText(QApplication::translate("MainWindow", "Nbr Core :", nullptr));
        UTF8_CheckBox->setText(QApplication::translate("MainWindow", "UTF-8", nullptr));
        CharSpe_CheckBox->setText(QApplication::translate("MainWindow", "Char. Sp\303\251cial", nullptr));
        AttaqueDico_CheckBox->setText(QApplication::translate("MainWindow", "Attaque dico", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Sortie et r\303\251sultat", nullptr));
        label->setText(QApplication::translate("MainWindow", "MDP:", nullptr));
        menuFichier->setTitle(QApplication::translate("MainWindow", "Fichier", nullptr));
        menuAide->setTitle(QApplication::translate("MainWindow", "Aide", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
