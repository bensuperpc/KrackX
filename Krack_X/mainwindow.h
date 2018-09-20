#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_actionDivers_2_triggered();

    void on_CoreUse_horizontalSlider_actionTriggered(int action);

    void on_CoreUse_horizontalSlider_sliderMoved(int position);

    void on_CoreUse_horizontalSlider_sliderPressed();

    void on_UTF8_CheckBox_clicked();

    void on_CharSpe_CheckBox_clicked();

    void on_AttaqueDico_CheckBox_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
