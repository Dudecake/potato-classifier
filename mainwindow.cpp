#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QPainter>
#include <QtDebug>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), loadFuture(std::future<void>())
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
        //loadFuture = DefaultThreadPool::submitJob(&MainWindow::handleImage, this, fileName);
        handleImage(this, fileName);
    }
}

void MainWindow::saveClassification()
{
    QString fileName = QFileDialog::getSaveFileName(this, QStringLiteral("Save classification"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation), QStringLiteral("PNG Files (*.png)"));
    if (!fileName.isEmpty())
    {
        loadFuture.get();
        image.save(fileName);
    }
}

void MainWindow::handleImage(MainWindow *window, QString &fileName)
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    window->image = QPixmap(fileName);
    window->ui->frame->setPixmap(window->image.scaled(window->ui->frame->width(), window->ui->frame->height(), Qt::KeepAspectRatio));
    DSImage::ImagePNG<float> imageMat(fileName.toStdString(), true);

    unsigned int patchSize = 14;
    unsigned int imageX = imageMat.rows.count();
    unsigned int imageY = imageMat.cols.count() / imageMat.getChannelCount();
    DSLib::Matrix<DSImage::ImagePNG<float>> resultMat;
    std::vector<DSImage::ImagePNG<float>> images((imageX / patchSize) * (imageY / patchSize));
    std::vector<ThreadPool::TaskFuture<DSImage::ImagePNG<float>>> futures;
    auto start = high_resolution_clock::now();

    for (unsigned int i = 0; i < (imageX / patchSize) * (imageY / patchSize); i++)
    {
        //images.push_back(getSubSection(imageMat, patchSize * (i % (imageX / patchSize)), patchSize, patchSize * (i / (imageY / patchSize)), patchSize));
        futures.push_back(window->threadPool.submit([](DSImage::ImagePNG<float> &imageMat, const unsigned int i, const unsigned int imageX, const unsigned int imageY, const unsigned int &patchSize){
            return MainWindow::getSubSection(imageMat, patchSize * (i % (imageX / patchSize)), patchSize, patchSize * (i / (imageY / patchSize)), patchSize);
        }, imageMat, i, imageX, imageY, patchSize));
    }
    std::transform(futures.begin(), futures.end(), images.begin(), [](ThreadPool::TaskFuture<DSImage::ImagePNG<float>> &f) -> DSImage::ImagePNG<float> { return f.get(); });
    futures.clear();
    qInfo() << "Loaded image in" << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << "ms";
    //pipeline.setBatchSize(static_cast<unsigned int>(images.size()));
    DSLib::Table<> modelData = (DSTypes::ctFeature | DSLib::Matrix<DSImage::ImagePNG<float>>(static_cast<unsigned int>(images.size()), 1u, images));
    try {
        DSLib::Table<unsigned int> valScore = window->pipeline.apply(modelData);
        //image = QPixmap(QSize(static_cast<int>(imageX), static_cast<int>(imageY)));
        //window->image.fill(Qt::black);
        QPainter painter;
        painter.begin(&window->image);
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
            int rectX = static_cast<int>((i % (static_cast<unsigned int>(window->image.width()) / patchSize)) * patchSize);
            int rectY = static_cast<int>(((i + 1) / (static_cast<unsigned int>(window->image.height()) / patchSize)) * patchSize);
            painter.drawRect(rectX, rectY, static_cast<int>(patchSize), static_cast<int>(patchSize));
        }
        window->ui->frame->setPixmap(window->image);
    }
    catch (const std::exception &ex)
    {
        qWarning() << ex.what();
        QMessageBox::critical(window, "Error", ex.what(), QMessageBox::Ok);
    }
}

MainWindow::~MainWindow()
{

}
