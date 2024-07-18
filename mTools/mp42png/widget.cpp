#include "widget.h"
#include "ui_widget.h"
#include <QMessageBox>
#include <QFileDialog>
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include <QDebug>

using namespace cv;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->pushButton_2->setEnabled(0);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_pushButton_pressed()
{
    QString currentFilter = "(*.mp4)";
    filePath = QFileDialog::getOpenFileName(this,
                                                  "请选择一个视频文件","",
                                                  "(*.mp4);;(*.flv);;(*.webm);;(*.mov)",
                                                  &currentFilter,QFileDialog::HideNameFilterDetails);
    if(filePath=="")
        return;
    ui->pushButton_2->setEnabled(1);
    ui->pushButton->setEnabled(0);
}


void Widget::on_pushButton_2_pressed()
{
    savePath = QFileDialog::getExistingDirectory(this,"请选择一个文件夹,路径请务必为全英文。");
    if(savePath=="")
        return;
    qDebug()<<savePath;
    //open videoFile
    VideoCapture videoFile;
    videoFile.open(filePath.toStdString());
    if(!videoFile.isOpened())
    {
        QMessageBox::information(nullptr,"提示","打开视频文件失败，请重试。");
        return;
    }
    //save to imgs
    Mat oneFrame;
    for(int z=0;z<videoFile.get(CAP_PROP_FRAME_COUNT);z++)
    {
        videoFile>>oneFrame;
        cvtColor(oneFrame,oneFrame,COLOR_BGR2GRAY);
        QString p = savePath + QString("/") + QString::number(z) + QString(".bmp");
        imwrite(p.toStdString(),oneFrame);
    }
    videoFile.release();
    QMessageBox::information(nullptr,"提示","转化为图像序列成功。");
    ui->pushButton_2->setEnabled(0);
    ui->pushButton->setEnabled(1);
}

