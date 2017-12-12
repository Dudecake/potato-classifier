#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QPainter>
#include <QtDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->modelLoadButton, &QToolButton::pressed, this, &MainWindow::loadDDSLModel);
    connect(ui->classifyButton, &QToolButton::pressed, this, &MainWindow::classify);
    connect(ui->saveButton, &QToolButton::pressed, this, &MainWindow::saveClassification);

    #ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
    caffe::Caffe::set_multiprocess(true);
    #else
    DSLib::Matrix<int> gpus(DSTypes::dtInt32 | 0);
    setCaffeGPUs(gpus);
    #endif
}

void MainWindow::loadDDSLModel()
{
    QString dirName = QFileDialog::getExistingDirectory(this, QStringLiteral("Open trained model"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation));
    if (!dirName.isEmpty())
    {
        QDir dir(dirName);
        QString modelName, solverName, pipelineName;
        QDir::setCurrent(dirName);
        for (QString file : dir.entryList())
        {
            if (file.endsWith("prototxt"))
            {
                if (file.contains("model"))
                    modelName = file;
                else if(file.contains("solver"))
                    solverName = file;
            }
            else if (file.endsWith("caffemodel.ddsl"))
                pipelineName = file;
        }
        // TODO: Load ddsl model
        pipeline = +DSModel::Caffe<float>((DSTypes::dtFloat | 0.f || 1.f), modelName.toStdString(), solverName.toStdString(), gpus);
        pipeline.read(pipelineName.toStdString());
    }
}

void MainWindow::classify()
{
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("Open image to classify"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation), QStringLiteral("PNG Files (*.png)"));

    if (!fileName.isEmpty())
    {
        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::milliseconds;

        DSImage::ImagePNG<float> imageMat(fileName.toStdString(), true);
        DSLib::Matrix<DSImage::ImagePNG<float>> resultMat;
        std::vector<DSImage::ImagePNG<float>> images;
//        std::vector<std::shared_future<DSImage::ImagePNG<float>>> futures;
        unsigned int patchSize = 14;
        unsigned int imageX = imageMat.rows.count();
        unsigned int imageY = imageMat.cols.count() / imageMat.getChannelCount();
        //resultMat | imageMat(0, patchSize, 0, patchSize);
        auto start = high_resolution_clock::now();

        for (unsigned int i = 0; i < (imageX / patchSize) * (imageY / patchSize); i++)
        {
            images.push_back(getSubSection(imageMat, patchSize * (i % (imageX / patchSize)), patchSize, patchSize * (i / (imageY / patchSize)), patchSize));
//            futures.push_back(std::async(std::launch::async, [&imageMat, &patchSize, &i, &imageX, &imageY](){
//                return getSubSection(imageMat, patchSize * (i % (imageX / patchSize)), patchSize, patchSize * (i / (imageY / patchSize)), patchSize);
//            }));
//            if (futures.size() >= 10)
//            {
//                std::transform(futures.begin(), futures.end(), images.begin(), [](std::shared_future<DSImage::ImagePNG<float>> &f) -> DSImage::ImagePNG<float> { return f.get(); });
//                futures.clear();
//            }
        }
//        std::transform(futures.begin(), futures.end(), images.begin(), [](std::shared_future<DSImage::ImagePNG<float>> &f) -> DSImage::ImagePNG<float> { return f.get(); });
//        futures.clear();
        qInfo() << "Loaded image in" << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << "ms";
        //pipeline.setBatchSize(static_cast<unsigned int>(images.size()));
        DSLib::Table<> modelData = (DSTypes::ctFeature | DSLib::Matrix<DSImage::ImagePNG<float>>(static_cast<unsigned int>(images.size()), 1u, images));
        DSLib::Table<unsigned int> valScore = pipeline.apply(modelData);
//        for(unsigned int idx = 0; idx < imageX / patchSize; idx++)
//            for(unsigned int idy = 0; idy < imageY / patchSize; idy++)
//                images.push_back(imageMat(patchSize * idy, patchSize, patchSize * idx, patchSize));
        image = QPixmap(QSize(static_cast<int>(imageX), static_cast<int>(imageY)));
        image.fill(Qt::black);
        QPainter painter;
        painter.begin(&image);
        DSLib::Matrix<float> results = valScore.findMatrix(DSTypes::ctResult, DSTypes::dtFloat)->data();
        for (unsigned int i = 0; i < results.data().size() / 2; i++)
        {
            switch (static_cast<int>(results.data().at(i)))
            {
                case 0:
                    painter.setBrush(Qt::red);
                    break;
                case 1:
                    painter.setBrush(Qt::black);
                    break;
                default:
                    break;
            }
            painter.drawRect(static_cast<int>((i % (imageX / patchSize)) * patchSize), static_cast<int>((i / (imageY / patchSize)) * patchSize), static_cast<int>(patchSize), static_cast<int>(patchSize));
        }
        ui->frame->setPixmap(image);
    }
}

void MainWindow::saveClassification()
{
    QString fileName = QFileDialog::getSaveFileName(this, QStringLiteral("Save classification"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation), QStringLiteral("PNG Files (*.png)"));
    if (!fileName.isEmpty())
    {
        image.save(fileName);
    }
}

MainWindow::~MainWindow()
{

}
