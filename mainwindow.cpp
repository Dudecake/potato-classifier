#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QPainter>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->modelLoadButton, &QToolButton::pressed, this, &MainWindow::loadDDSLModel);
    connect(ui->classifyButton, &QToolButton::pressed, this, &MainWindow::classify);
    connect(ui->saveButton, &QToolButton::pressed, this, &MainWindow::saveClassification);
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
        pipeline.read(pipelineName.toStdString());
    }
}

void MainWindow::classify()
{
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("Open image to classify"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation), QStringLiteral("PNG Files (*.png)"));

    if (!fileName.isEmpty())
    {
        DSImage::ImagePNG<float> imageMat(fileName.toStdString());
        QVector<DSImage::ImagePNG<float>> images;
        unsigned int patchSize = 16;
        for (unsigned int i = 0; i < (imageMat.rows.count() / patchSize) * (imageMat.cols.count() / patchSize); i++)
        {
            images.append(imageMat(patchSize * (i % (imageMat.rows.count() / patchSize)), patchSize, patchSize * (i / (imageMat.cols.count() / patchSize)), patchSize));
        }
        DSLib::Table<> modelData = (DSTypes::ctTarget | DSLib::Matrix<DSImage::ImagePNG<float>>(static_cast<unsigned int>(images.size()), 1u, images.toStdVector()));
        DSLib::Matrix<int> valScore = pipeline.train(modelData);
//        for(unsigned int idx = 0; idx < imageMat.rows.count() / patchSize; idx++)
//            for(unsigned int idy = 0; idy < imageMat.cols.count() / patchSize; idy++)
//                images.push_back(imageMat(patchSize * idy, patchSize, patchSize * idx, patchSize));
        image = QPixmap(QSize(static_cast<int>(imageMat.rows.count()), static_cast<int>(imageMat.cols.count())));
        image.fill(Qt::black);
        QPainter painter;
        painter.begin(&image);
        for (unsigned int i = 0; i < valScore.data().size() / 2; i++)
        {
            switch (valScore.data().at(i))
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
            painter.drawRect(static_cast<int>((i % (imageMat.rows.count() / patchSize)) * patchSize), static_cast<int>((i / (imageMat.cols.count() / patchSize)) * patchSize), static_cast<int>(patchSize), static_cast<int>(patchSize));
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
