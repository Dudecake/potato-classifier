#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <ddsl.hpp>

#include "threadpool.hpp"

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
        std::string sayHi(std::string name) const
        {
            return std::string("Hi ").append(name);
        }
        ~MainWindow();

    public slots:
        void loadDDSLModel();
        void classify();
        void saveClassification();

    private:
        Ui::MainWindow *ui;
        ThreadPool::TaskFuture<void> loadFuture;
        ThreadPool threadPool;
        DSLib::Matrix<int> gpus;
        DSModel::Caffe<float> pipeline;
        QPixmap image;
        static inline DSImage::ImagePNG<float> getSubSection(DSImage::ImagePNG<float> &image, const unsigned int x, const unsigned int width, const unsigned int y, const unsigned int height)
        {
            return DSImage::ImagePNG<float>(std::string(std::tmpnam(nullptr)) + ".png",
                                            (image.getChannel(0).dup()(x, width, y, height).dup() |
                                             image.getChannel(1).dup()(x, width, y, height).dup() |
                                             image.getChannel(2).dup()(x, width, y, height).dup()),
                                            DSTypes::ImageType::itRGB8Planar);
        }
        static inline void getSubSection(DSImage::ImagePNG<float> &res, DSImage::ImagePNG<float> &image, unsigned int x, unsigned int width, unsigned int y, unsigned int height)
        {
            res = DSImage::ImagePNG<float>(std::string(std::tmpnam(nullptr)) + ".png",
                                            (image.getChannel(0).dup()(x, width, y, height).dup() |
                                             image.getChannel(1).dup()(x, width, y, height).dup() |
                                             image.getChannel(2).dup()(x, width, y, height).dup()),
                                            DSTypes::ImageType::itRGB8Planar);
        }
        static void handleImage(MainWindow *window, QString &fileName);
};

#endif // MAINWINDOW_H
