#pragma once

#include <QAbstractTableModel>
#include <QObject>

    class TableModel : public QAbstractTableModel {
    Q_OBJECT
    enum TableRoles { TableDataRole = Qt::UserRole + 1, HeadingRole };
public:
    explicit TableModel(QObject *parent = nullptr) : QAbstractTableModel(parent) {
        this->init();
    }
    void clear()
    {
        this->table.clear();
        this->table.shrink_to_fit();
        this->init();
    }
    void init()
    {
        table.append({
                      "Iter. NÂ°",
                      "Code",
                      "JAMCRC value",
                      });
    }

    int rowCount(const QModelIndex & = QModelIndex()) const override {
        return table.size();
    }

    int columnCount(const QModelIndex & = QModelIndex()) const override {
        return table.at(0).size();
    }

    QVariant data(const QModelIndex &index, int role) const override {
        switch (role) {
        case TableDataRole: {
            return table.at(index.row()).at(index.column());
        }
        case HeadingRole: {
            return index.row() == 0;
        }
        default: break;
        }
        return QVariant();
    }
    QHash<int, QByteArray> roleNames() const override {
        QHash<int, QByteArray> roles;
        roles[TableDataRole] = "tabledata";
        roles[HeadingRole] = "heading";
        return roles;
    }

        Q_INVOKABLE void addPerson() {
        beginInsertRows(QModelIndex(), rowCount(), rowCount());
        table.append({
                      "Marc",
                      "Fonz",
                      "25",
                      });
        endInsertRows();
    }

        Q_INVOKABLE void addPerson(const QVector<QString> &vect) {
        beginInsertRows(QModelIndex(), rowCount(), rowCount());
        table.append(vect);
        endInsertRows();
    }

        Q_INVOKABLE void addPerson(const std::vector<QString> &vect) {
            beginInsertRows(QModelIndex(), rowCount(), rowCount());
#if QT_VERSION <= QT_VERSION_CHECK(5,14,0)
            table.append(QVector<QString>::fromStdVector(vect));
#else
            table.append(QVector<QString>(vect.begin(), vect.end()));
#endif
            endInsertRows();
        }
private:
    QVector<QVector<QString>> table;
};
