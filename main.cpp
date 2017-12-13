#include <QApplication>
#include <QCommandLineParser>

#include "mainwindow.h"
#include <sched.h>

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
    cpu_set_t set;
    CPU_ZERO(&set);
    if (std::thread::hardware_concurrency() != 0)
        for (unsigned int i = 0; i < std::thread::hardware_concurrency(); i++)
            CPU_SET(i, &set);
    else
    {
        CPU_SET(0, &set);
        CPU_SET(1, &set);
        CPU_SET(2, &set);
        CPU_SET(3, &set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    MainWindow w;
    w.show();
    return a.exec();
}
