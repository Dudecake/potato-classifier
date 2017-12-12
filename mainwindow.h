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
        DSLib::Matrix<int> gpus;
        DSModel::Caffe<float> pipeline;
        QPixmap image;
        static inline DSImage::ImagePNG<float> getSubSection(DSImage::ImagePNG<float> &image, unsigned int x, unsigned int width, unsigned int y, unsigned int height)
        {
            return DSImage::ImagePNG<float>(std::string(std::tmpnam(nullptr)) + ".png",
                                            (image.getChannel(0).dup()(x, width, y, height).dup() |
                                             image.getChannel(1).dup()(x, width, y, height).dup() |
                                             image.getChannel(2).dup()(x, width, y, height).dup()),
                                            DSTypes::ImageType::itRGB8Planar);
        }
        static inline DSImage::ImagePNG<float>* getSubSectionPtr(DSImage::ImagePNG<float> &image, unsigned int x, unsigned int width, unsigned int y, unsigned int height)
        {
            return new DSImage::ImagePNG<float>(std::string(std::tmpnam(nullptr)) + ".png",
                                            (image.getChannel(0).dup()(x, width, y, height).dup() |
                                             image.getChannel(1).dup()(x, width, y, height).dup() |
                                             image.getChannel(2).dup()(x, width, y, height).dup()),
                                            DSTypes::ImageType::itRGB8Planar);
        }
};

#endif // MAINWINDOW_H
