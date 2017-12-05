#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <ddsl.hpp>

namespace Ui
{
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    public:
        explicit MainWindow(QWidget *parent = nullptr);
        static void setupGPUs()
        {
            #ifdef CPU_ONLY
            caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
            caffe::Caffe::set_multiprocess(true);
            DSLib::Matrix<int> gpus;// = (DSTypes::dtInt32 | 0);
            #else
            DSLib::Matrix<int> gpus(DSTypes::dtInt32 | 0);
            setCaffeGPUs(gpus);
            #endif
        }
        ~MainWindow();

    public slots:
        void loadDDSLModel();
        void classify();
        void saveClassification();

    private:
        Ui::MainWindow *ui;
        DSModel::Caffe<float> pipeline;
        QPixmap image;
};

#endif // MAINWINDOW_H
