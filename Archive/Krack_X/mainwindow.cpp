#include <QtDebug>
#include <QDate>
#include <QMessageBox>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "core.h"
//#include<stdio.h> //    //system("google-chrome");
//#include<iostream>
//#include "debug.h"

static CoreProcessing CoreApp;


MainWindow::MainWindow(QWidget *parent) :

    QMainWindow(parent),

    ui(new Ui::MainWindow)
{ 
    ui->setupUi(this);
    ui->CoreUse_Label->setText("Nbr Core : " + QString::fromStdString(std::to_string(CoreApp.UseCore)));
    ui->CoreUse_horizontalSlider->setSliderPosition(CoreApp.UseCore);
    //connect(object_ptr, &OutputConsole::variableChanged, [=](int i){
     //   ui->OutputConsole->setText("Number = " + QString::number(i));
    //});
    //connect(CoreApp.OuputConsole,SIGNAL(valueChanged(int)),ui->OutputConsole,SLOT(),);


    //connect(CoreApp.OuputConsole, SIGNAL(changeText(QString)), ui->OutputConsole, SLOT(labelChangeText(QString)));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    qDebug() <<"CoreApp.UseCore" + QString::fromStdString(std::to_string(CoreApp.UseCore));
    qDebug() << "Bouton OK";
    qDebug() << "===========================================================";
    QString CommandLaunchDate = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    ui->OutputConsole->setText(ui->OutputConsole->toPlainText()+CommandLaunchDate + " " +"Lancement de google-chrome"+"\n");
    QString* OuputConsole = &CoreApp.OutputConsole;
    //ui->OutputConsole->setText(OuputConsole);


    CoreApp.exec("google-chrome");
    qDebug() << QString::fromStdString("");
    qDebug() << "============================================================";
    ui->progressBar->setValue(ui->progressBar->value()+1);
}


void MainWindow::on_actionDivers_2_triggered()
{
    const std::string Compilator_Version = std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
    const QString CPP_Info = QString::fromStdString(std::to_string(__cplusplus));

    const QString QCompilator_Version = "Compiler Version :" + QString::fromStdString(Compilator_Version);
    //QString sDate = QDateTime::currentDateTime().toString("dddd dd MMMM yyyy hh:mm:ss.zzz");
    const QString sDate = QDateTime::currentDateTime().toString("dd/MM/yyyy");
    QMessageBox msgBox;

    msgBox.setText("<b>Build informations</b>");
    const QString Message = QString("Build by: ") + QCompilator_Version + "\n" + "C++ version: " + CPP_Info + "\n" + "Build date :" + sDate;
    msgBox.setInformativeText(Message);
    msgBox.exec();
}


void MainWindow::on_UTF8_CheckBox_clicked()
{
    CoreApp.UTF8_Password = ui->UTF8_CheckBox->isTristate();
    qDebug() << "UTF-8 Bouton";
    qDebug() << "UTF-8 Bouton";
}

void MainWindow::on_CharSpe_CheckBox_clicked()
{
    CoreApp.CharSpe_Password = ui->CharSpe_CheckBox->isTristate();
    qDebug() << "UTF-8 Bouton";
}

void MainWindow::on_AttaqueDico_CheckBox_clicked()
{
    CoreApp.AttacByDico = ui->AttaqueDico_CheckBox->isTristate();
}

void MainWindow::on_CoreUse_horizontalSlider_valueChanged(int value)
{
    qDebug() <<"CoreUse_horizontalSlider_sliderPressed :" + QString::fromStdString(std::to_string(ui->CoreUse_horizontalSlider->value()));
    ui->CoreUse_Label->setText("Nbr Core : " + QString::fromStdString(std::to_string(ui->CoreUse_horizontalSlider->value())));
    CoreApp.UseCore = (unsigned)ui->CoreUse_horizontalSlider->value();
}

/* Sources */
//http://www.electro-info.ovh/index.php?id=213
//http://doc.qt.io/qt-5/qdatetime.html
