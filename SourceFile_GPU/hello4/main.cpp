#include <QApplication>
#include <QWidget>
#include <QDebug>
#include "algorithm.h"

int main(int argc, char *argv[])
{
    QApplication a(argc,argv);

    QWidget w;
    w.setWindowTitle("I am title");
    w.resize(300,200);
    w.move(0,0);
    w.setCursor(Qt::UpArrowCursor);
    w.show();

    test_cuda();

    return a.exec();
}
