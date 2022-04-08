#include "gta_sa_ui.h"

GTA_SA_UI::GTA_SA_UI(QObject *parent)
    : QObject{parent}
{

}

void GTA_SA_UI::setMinRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__  << ": " << value << std::endl;
    if (value == _minRangeValue)
        return;
    _minRangeValue = value;
    gta_sa.min_range = value;
    emit minRangeValueChanged(value);
}

void GTA_SA_UI::setMaxRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__  << ": " << value << std::endl;
    if (value == _maxRangeValue)
        return;
    _maxRangeValue = value;
    gta_sa.max_range = value;
    emit maxRangeValueChanged(value);
}

void GTA_SA_UI::setNbrThreadValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__  << ": " << value << std::endl;
    if (value == _nbrThreadValue)
        return;
    _nbrThreadValue = value;
    emit nbrThreadValueChanged(value);
}

void GTA_SA_UI::setButtonValue(QString value) {
    std::cout << __FUNCTION_NAME__  << ": " << value.toStdString() << std::endl;
    if (value == _buttonValue)
        return;
    _buttonValue = value;
    emit buttonValueChanged(value);
}

void GTA_SA_UI::runOpThread()
{


    // Clear old data
    this->gta_sa.clear();
    this->tableModel.clear();

    // Launch operation
    this->gta_sa.runner();

    // Store results in TableView Data
    for(const auto &result : this->gta_sa.results)
    {
        QVector<QString> vect = {QString::number(std::get<0>(result)), QString::fromStdString(std::get<1>(result)),QString::number(std::get<2>(result))};
        this->tableModel.addPerson(vect);
    }
    this->setButtonValue("  Launch Bruteforce   ");
}

void GTA_SA_UI::runOp()
{
    this->setButtonValue("Bruteforce in progress");
    std::thread runner_thread(&GTA_SA_UI::runOpThread, this);
    runner_thread.detach();
    // runner_thread.join();
}
