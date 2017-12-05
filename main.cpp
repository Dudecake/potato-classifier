#include <QApplication>
#include <QCommandLineParser>

#include "mainwindow.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCoreApplication::setOrganizationDomain("nhl.nl");
    QCoreApplication::setOrganizationName("NHL HogeSchool");
    QCoreApplication::setApplicationName("Potato-Classifier");
    QCoreApplication::setApplicationVersion("1.0.0");
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    parser.process(a);

    MainWindow::setupGPUs();
    MainWindow w;
    w.show();
    return a.exec();
}
