#include "widget.h"
#include "ui_widget.h"

#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QMouseEvent>
#include <QKeyEvent>
#include <string>
#include <math.h>
#include <QString>
//concurrency
#include <vector>
#include <thread>

using namespace std;

typedef unsigned char uchar;

//show 3D
#include "imagecube.h"
#include "viewport.h"
#include "canvas.h"
#include "scene.h"
#define Pi 3.14159265358979323846
//int canvasWidth = 830; //must be the same as label_4->Width
//int canvasHeight = 531; //must be the same as label_4->Height
int canvasWidth = 960;
int canvasHeight = 540;
Canvas cvs(canvasWidth,canvasHeight);
Scene scene; //this is initialized in "void Widget::on_loadFileButton_clicked()"
double vptWidth=0.82, vptHeight=0.52;
Viewport viewport; //this is initialized in "void Widget::show3D()"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->label_1->setStyleSheet("QLabel{background-color:rgb(0,0,0);}");ui->label_1->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->label_2->setStyleSheet("QLabel{background-color:rgb(0,0,0);}");ui->label_2->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->label_3->setStyleSheet("QLabel{background-color:rgb(0,0,0);}");ui->label_3->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->label_4->setStyleSheet("QLabel{background-color:rgb(0,0,0);}");ui->label_4->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    //show viewport location
    ui->label_alpha->setText(QString("α=")+QString::number(alpha,'f',2));
    ui->label_beta->setText(QString("β=")+QString::number(beta,'f',2));
    ui->label_rho->setText(QString("ρ=")+QString::number(rho,'f',2));
    //disable all elements
    ui->doubleSpinBox->setEnabled(false);
    ui->doubleSpinBox_2->setEnabled(false);
    ui->doubleSpinBox_3->setEnabled(false);
    ui->doubleSpinBox_4->setEnabled(false);
    ui->doubleSpinBox_5->setEnabled(false);
    ui->comboBox_SR->setEnabled(false);
    ui->checkBox->setEnabled(false);
    ui->comboBox_rayTracing->setEnabled(false);
    ui->spinBox->setEnabled(false);
    ui->checkBoxShowOriginalSize_2->setEnabled(false);
    ui->lowThreshold->setEnabled(false);
    ui->highThreshold->setEnabled(false);
}
Widget::~Widget()
{
    delete ui;
}

//choose file and load file into array3D
void Widget::chooseFile()
{
    this->imagePaths = QFileDialog::getOpenFileNames(nullptr,
                                                     "选择图片",
                                                     "",
                                                     "Images (*.png *.xpm *.jpg *.bmp)");
}
void mySort(QStringList *qsList)//pop sort
{
    int len = qsList->length();
    for(int i=len-1;i>0;i--){
        for(int j=0;j<i;j++){
            int a = (*qsList)[j].split(".")[0].toInt();
            int b = (*qsList)[j+1].split(".")[0].toInt();
            if(a>b){
                QString temp=(*qsList)[j];
                (*qsList)[j] = (*qsList)[j+1];
                (*qsList)[j+1] = temp;
            }
        }
    }
}
void Widget::refresh()
{
    this->img3D.destroy();
    this->xyPlane.destroy();this->xzPlane.destroy();this->yzPlane.destroy();
    allowShowAll = false;
}
void Widget::loadVideo2Array3D()
{
    //sort files
    mySort(&this->imagePaths);
    //set size
    QString img0Name = this->imagePaths[0];
    QImage img0 = QImage(img0Name);
    this->size_x=img0.width(); this->size_y=img0.height(); this->size_z=this->imagePaths.length();
    qDebug()<<"size_x:"<<this->size_x<<", size_y:"<<this->size_y<<", size_z:"<<this->size_z<<Qt::endl;
    //allocate memory
    this->img3D.malloc(this->size_x,this->size_y,this->size_z);
    this->xyPlane.malloc(img3D.get_size_x(),img3D.get_size_y());
    this->xzPlane.malloc(img3D.get_size_x(),img3D.get_size_z());
    this->yzPlane.malloc(img3D.get_size_z(),img3D.get_size_y());
    //save imgs into img3D
    for(int z=0;z<this->size_z;z++)
    {
        QString imgName = this->imagePaths[z];
        QImage img = QImage(imgName);
        for(int x=0;x<this->size_x;x++)
        {
            for(int y=0;y<this->size_y;y++)
            {
                QRgb p = img.pixel(x,y);
                uchar g = qGray(p);
                this->img3D.set(x,y,z,g);
            }
        }
    }
    //show success
    QMessageBox::information(this,"提示","图像序列加载成功。");
}
void Widget::on_loadFileButton_clicked()
{
    //choose file
    this->chooseFile();
    if(this->imagePaths.length()==0)
        return;
    //enable all elements
    ui->doubleSpinBox->setEnabled(true);
    ui->doubleSpinBox_2->setEnabled(true);
    ui->doubleSpinBox_3->setEnabled(true);
    ui->doubleSpinBox_4->setEnabled(true);
    ui->doubleSpinBox_5->setEnabled(true);
    ui->comboBox_SR->setEnabled(true);
    ui->checkBox->setEnabled(true);
    ui->comboBox_rayTracing->setEnabled(true);
    ui->spinBox->setEnabled(true);
    ui->checkBoxShowOriginalSize_2->setEnabled(true);
    ui->lowThreshold->setEnabled(true);
    ui->highThreshold->setEnabled(true);
    ui->checkBox_xOy->setEnabled(true);
    ui->checkBox_xOz->setEnabled(true);
    ui->checkBox_yOz->setEnabled(true);
    //get filePath
    QFileInfo fileInfo(imagePaths[0]);
    this->filePath = fileInfo.path();
    qDebug()<<"filePath: "<<this->filePath<<Qt::endl;
    //refresh and load
    refresh();
    this->loadVideo2Array3D();
    // display size initiate
    xyDisplay_X = int(ui->doubleSpinBox->value()*size_x); xyDisplay_Y = int(ui->doubleSpinBox_2->value()*size_y);
    xzDisplay_X = int(ui->doubleSpinBox->value()*size_x); xzDisplay_Y = int(ui->doubleSpinBox_3->value()*size_z);
    yzDisplay_X = int(ui->doubleSpinBox_3->value()*size_z); yzDisplay_Y = int(ui->doubleSpinBox_2->value()*size_y);
    // display method initiate
    switch (ui->comboBox_SR->currentIndex()) {
    case 0:
        this->interpolationMethod = Nearest;
        break;
    case 1:
        this->interpolationMethod = Bilinear;
        break;
    default:
        break;
    }
    //reset spinbox scrollbar maxnumber
    ui->horizontalScrollBar_X->setValue(1);//set to max to couple with the 3D view (NO)
    ui->spinBox_1->setMaximum(this->size_z);
    ui->horizontalScrollBar_X->setMaximum(this->size_z);
    ui->horizontalScrollBar_Y->setValue(1);//set to couple with the 3D view
    ui->spinBox_2->setMaximum(this->size_y);
    ui->horizontalScrollBar_Y->setMaximum(this->size_y);
    ui->horizontalScrollBar_Z->setValue(1);//set to couple with the 3D view
    ui->spinBox_3->setMaximum(this->size_x);
    ui->horizontalScrollBar_Z->setMaximum(this->size_x);
    allowShowAll = true;
    //showAll
    this->showAll();
    //display 3D
    scene.addCube(&this->img3D, cb_scaleX, cb_scaleY, cb_scaleZ);
    ui->horizontalSlider_pho->setValue((int)(rho-rhoMin)/(rhoMax-rhoMin)*100);

    ui->doubleSpinBox->setValue(1.00);
    ui->doubleSpinBox_2->setValue(1.00);
    ui->doubleSpinBox_3->setValue(1.00);
    this->show3D();
    //resizeEvent
    programmeIsOn = true;
}

//show all
void Widget::showAll()
{
    if(allowShowAll)
    {
        this->xyShow(this->xyPlaneIdx);
        this->xzShow(this->xzPlaneIdx);
        this->yzShow(this->yzPlaneIdx);
    }
}
//label show
void Widget::labelShow(QLabel *lb, QPixmap *pxm)
{
//    //三幅图等比例放到最大
//    int pw=pxm->width(), ph=pxm->height();
//    int lbw=lb->width(), lbh=lb->height();
    double scale1 = max( (double)xyDisplay_X/(double)ui->label_1->width(), (double)xyDisplay_Y/(double)ui->label_1->height() );
    double scale2 = max( (double)xzDisplay_X/(double)ui->label_2->width(), (double)xzDisplay_Y/(double)ui->label_2->height() );
    double scale3 = max( (double)yzDisplay_X/(double)ui->label_3->width(), (double)yzDisplay_Y/(double)ui->label_3->height() );
    double maxScale = max(scale1,max(scale2,scale3));

    *pxm = pxm->scaled(int(pxm->width()/maxScale),int(pxm->height()/maxScale), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    lb->setPixmap(*pxm);
}
//xy plane
void Widget::xyShow(int z)
{
    //Load the plane from img3D
    xyPlane.loadFromArray3D(&img3D,XY,z);
    //interpolation
    Array2D<uchar> xyPlaneDisplay(this->xyDisplay_X, this->xyDisplay_Y);
    switch (this->interpolationMethod) {
    case Nearest:
        xyPlane.nearest(&xyPlaneDisplay);
        break;
    case Bilinear:
        xyPlane.bilinear(&xyPlaneDisplay);
        break;
    default:
        break;
    }
    //draw locating Lines
    if(showLocation){
        xyPlaneDisplay.drawLocatingLines(
            (int)((double)this->yzPlaneIdx * (double)this->xyDisplay_X / this->size_x),
            (int)((double)this->xzPlaneIdx * (double)this->xyDisplay_Y / this->size_y),
            this->showLocationWIdth,
            this->showLocationColor);
    }
    //load pixel into qimg
    QImage qImg(xyPlaneDisplay.get_size_x(),xyPlaneDisplay.get_size_y(),QImage::Format_Grayscale8);
    for(int x=0;x<xyPlaneDisplay.get_size_x();x++)
    {
        for(int y=0;y<xyPlaneDisplay.get_size_y();y++)
        {
            uchar a=xyPlaneDisplay.at(x,y);
            a = a * (a>=transparentThreshold_low) * (a<=transparentThreshold_high);
            qImg.setPixel(x,y,qRgb(a,a,a));
        }
    }
    fitpixmapXOY = QPixmap::fromImage(qImg);
    //show.
    labelShow(ui->label_1, &fitpixmapXOY);
}
void Widget::zSliceChanged(int slice)
{
    this->xyPlaneIdx = slice-1;
    this->showAll();
    if(status[2]){
        scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,status,SliceMask);
        show3D();}
}
void Widget::on_spinBox_1_valueChanged(int arg1)
{
    if(ui->horizontalScrollBar_X->value()!=arg1)
    {
        ui->horizontalScrollBar_X->setValue(arg1);
        zSliceChanged(arg1);
    }
}
void Widget::on_horizontalScrollBar_X_valueChanged(int value)
{
    if(ui->spinBox_1->value()!=value)
    {
        ui->spinBox_1->setValue(value);
        zSliceChanged(value);
    }
}

//xz plane
void Widget::xzShow(int y)//
{
    //Load the plane from img3D
    xzPlane.loadFromArray3D(&img3D,XZ,y);
    //interpolation
    Array2D<uchar> xzPlaneDisplay(this->xzDisplay_X, this->xzDisplay_Y);
    switch (this->interpolationMethod) {
    case Nearest:
        xzPlane.nearest(&xzPlaneDisplay);
        break;
    case Bilinear:
        xzPlane.bilinear(&xzPlaneDisplay);
        break;
    default:
        break;
    }

    //draw locating Lines
    if(showLocation){
        xzPlaneDisplay.drawLocatingLines(
            (int)((double)this->yzPlaneIdx * (double)this->xzDisplay_X / this->size_x),
            (int)((double)this->xyPlaneIdx * (double)this->xzDisplay_Y / this->size_z),
            this->showLocationWIdth,
            this->showLocationColor);
    }
    //load pixel into qimg
    QImage qImg(xzPlaneDisplay.get_size_x(),xzPlaneDisplay.get_size_y(),QImage::Format_Grayscale8);
    for(int x=0;x<xzPlaneDisplay.get_size_x();x++)
    {
        for(int y=0;y<xzPlaneDisplay.get_size_y();y++)
        {
            uchar a=xzPlaneDisplay.at(x,y);
            a = a * (a>=transparentThreshold_low) * (a<=transparentThreshold_high);
            qImg.setPixel(x,y,qRgb(a,a,a));
        }
    }
    fitpixmapXOZ = QPixmap::fromImage(qImg);
    //show
    labelShow(ui->label_2, &fitpixmapXOZ);
}
void Widget::ySliceChanged(int slice)
{
    this->xzPlaneIdx = slice-1;
    this->showAll();
    if(status[1]){
        scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,status,SliceMask);
        show3D();}
}
void Widget::on_spinBox_2_valueChanged(int arg1)
{
    if(ui->horizontalScrollBar_Y->value()!=arg1)
    {
        ui->horizontalScrollBar_Y->setValue(arg1);
        ySliceChanged(arg1);
    }
}
void Widget::on_horizontalScrollBar_Y_valueChanged(int value)
{
    if(ui->spinBox_2->value()!=value)
    {
        ui->spinBox_2->setValue(value);
        ySliceChanged(value);
    }
}

//yz plane
void Widget::yzShow(int x)//
{
    //Load the plane from img3D
    yzPlane.loadFromArray3D(&img3D,YZ,x);
    //interpolation
    Array2D<uchar> yzPlaneDisplay(this->yzDisplay_X, this->yzDisplay_Y);
    switch (this->interpolationMethod) {
    case Nearest:
        yzPlane.nearest(&yzPlaneDisplay);
        break;
    case Bilinear:
        yzPlane.bilinear(&yzPlaneDisplay);
        break;
    default:
        break;
    }
    //draw locating Lines
    if(showLocation){
        yzPlaneDisplay.drawLocatingLines(
            (int)((double)this->xyPlaneIdx * (double)this->yzDisplay_X / this->size_z),
            (int)((double)this->xzPlaneIdx * (double)this->yzDisplay_Y / this->size_y),
            this->showLocationWIdth,
            this->showLocationColor);
    }
    //load pixel into qimg
    QImage qImg(yzPlaneDisplay.get_size_x(),yzPlaneDisplay.get_size_y(),QImage::Format_Grayscale8);
    for(int x=0;x<yzPlaneDisplay.get_size_x();x++)
    {
        for(int y=0;y<yzPlaneDisplay.get_size_y();y++)
        {
            uchar a=yzPlaneDisplay.at(x,y);
            a = a * (a>=transparentThreshold_low) * (a<=transparentThreshold_high);
            qImg.setPixel(x,y,qRgb(a,a,a));
        }
    }
    fitpixmapYOZ = QPixmap::fromImage(qImg);
    //show
    labelShow(ui->label_3, &fitpixmapYOZ);
}
void Widget::xSliceChanged(int slice)
{
    this->yzPlaneIdx = slice-1;
    this->showAll();
    if(status[0]){
        scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,status,SliceMask);
        show3D();}
}
void Widget::on_spinBox_3_valueChanged(int arg1)
{
    if(ui->horizontalScrollBar_Z->value()!=arg1)
    {
        ui->horizontalScrollBar_Z->setValue(arg1);
        xSliceChanged(arg1);
    }
}
void Widget::on_horizontalScrollBar_Z_valueChanged(int value)
{
    if(ui->spinBox_3->value()!=value)
    {
        ui->spinBox_3->setValue(value);
        xSliceChanged(value);
    }
}

//Interpolation
void Widget::on_comboBox_SR_currentIndexChanged(int index)
{
    switch (index) {
    case 0:
        this->interpolationMethod = Nearest;
        break;
    case 1:
        this->interpolationMethod = Bilinear;
        break;
    default:
        break;
    }
    showAll();
}

//show 3D
Vec3D canvas2Viewport(Canvas *cvs,Viewport *vpt,int i,int j,double beta)
{
    //get x and y of V in the scene
    double x = j * vpt->get_width() / (cvs->get_width()-1);
    x = x - vpt->get_width()/2;
    double y = i * vpt->get_height() / (cvs->get_height()-1);
    y = -1*y + vpt->get_height()/2;
    Vec3D cenV = vpt->cen_v(x,y,beta);
    //calculate D
    Vec3D caCen = vpt->ca_cen();
    Vec3D D = Vec3D::add(&caCen,&cenV);
    //D = D.unit();
    return D;
}
void interCubePlanes(Vec3D *O,Vec3D *D,int *numInterPlanes,int *interPlanesBool,double *ts,
                     double xoy_z_pos,double xoy_z_neg,
                     double xoz_y_pos,double xoz_y_neg,
                     double yoz_x_pos,double yoz_x_neg)
{
    //function of this subfunction: write result to numInterPlanes, interPlanesBool[6] and ts[6].
    double t,x,y,z;
    //parallel to any face
    if(D->get_x()==0 && D->get_y()==0)//XOY_pos,XOY_neg
    {
        *numInterPlanes=2;
        interPlanesBool[0]=1; interPlanesBool[1]=1;
        ts[0]=(xoy_z_pos - O->get_z()) / D->get_z();
        ts[1]=(xoy_z_neg - O->get_z()) / D->get_z();
    }
    else if(D->get_x()==0 && D->get_z()==0)//XOZ_pos,XOZ_neg
    {
        *numInterPlanes=2;
        interPlanesBool[2]=1; interPlanesBool[3]=1;
        ts[2]=(xoz_y_pos - O->get_y()) / D->get_y();
        ts[3]=(xoz_y_neg - O->get_y()) / D->get_y();
    }
    else if(D->get_y()==0 && D->get_z()==0)//YOZ_pos,YOZ_neg
    {
        *numInterPlanes=2;
        interPlanesBool[4]=1; interPlanesBool[5]=1;
        ts[4]=(yoz_x_pos - O->get_x()) / D->get_x();
        ts[5]=(yoz_x_neg - O->get_x()) / D->get_x();
    }
    else//not parallel, then decide planes
    {
        //XOY_pos
        if(D->get_z()!=0)
        {
            t = (xoy_z_pos - O->get_z())/D->get_z();
            x = O->get_x()+t*D->get_x(), y = O->get_y()+t*D->get_y();
            if (x>yoz_x_neg && x<yoz_x_pos && y>xoz_y_neg && y<xoz_y_pos){
                ts[0] = t;
                interPlanesBool[0] = 1;}
        }
        //XOY_neg
        if(D->get_z()!=0)
        {
            t = (xoy_z_neg - O->get_z())/D->get_z();
            x = O->get_x()+t*D->get_x(), y = O->get_y()+t*D->get_y();
            if (x>yoz_x_neg && x<yoz_x_pos && y>xoz_y_neg && y<xoz_y_pos){
                ts[1] = t;
                interPlanesBool[1] = 1;}
        }
        //XOZ_pos
        if(D->get_y()!=0)
        {
            t = (xoz_y_pos - O->get_y())/D->get_y();
            x = O->get_x()+t*D->get_x(), z = O->get_z()+t*D->get_z();
            if (x>yoz_x_neg && x<yoz_x_pos && z>xoy_z_neg && z<xoy_z_pos){
                ts[2] = t;
                interPlanesBool[2] = 1;}
        }
        //XOZ_neg
        if(D->get_y()!=0)
        {
            t = (xoz_y_neg - O->get_y())/D->get_y();
            x = O->get_x()+t*D->get_x(), z = O->get_z()+t*D->get_z();
            if (x>yoz_x_neg && x<yoz_x_pos && z>xoy_z_neg && z<xoy_z_pos){
                ts[3] = t;
                interPlanesBool[3] = 1;}
        }
        //YOZ_pos
        if(D->get_x()!=0)
        {
            t = (yoz_x_pos - O->get_x())/D->get_x();
            y = O->get_y()+t*D->get_y(), z = O->get_z()+t*D->get_z();
            if (y>xoz_y_neg && y<xoz_y_pos && z>xoy_z_neg && z<xoy_z_pos){
                ts[4] = t;
                interPlanesBool[4] = 1;}
        }
        //YOZ_neg
        if(D->get_x()!=0)
        {
            t = (yoz_x_neg - O->get_x())/D->get_x();
            y = O->get_y()+t*D->get_y(), z = O->get_z()+t*D->get_z();
            if (y>xoz_y_neg && y<xoz_y_pos && z>xoy_z_neg && z<xoy_z_pos){
                ts[5] = t;
                interPlanesBool[5] = 1;}
        }
    }
    for (int i=0;i<6;i++)
    {
        *numInterPlanes += interPlanesBool[i];
    }
}
uchar traceRay(Scene *scene,Vec3D *O,Vec3D *D,double tmin,double tmax,double tinc,
               enum PERSPECTIVE psp = EMAB,bool showFrame=true,double frameWidth=0.01,uchar frameColor=255,
               double sample_step=0.01, double absorption_rate=0.5,int transparentThreshold_low=0,int transparentThreshold_high=255,
               double transparent_reduce=1)
{
    uchar gLevel=0;

    ImageCube cube = scene->getCube();
    //decide the two interaction planes
    int numInterPlanes = 0;
    int interPlanesBool[6] = {0,0,0,0,0,0}; //[XOY_pos,XOY_neg,XOZ_pos,XOZ_neg,YOZ_pos,YOZ_neg]
    double ts[6] = {0,0,0,0,0,0};
    interCubePlanes(O,D,&numInterPlanes,interPlanesBool,ts,
                    cube.XOY_z_pos,cube.XOY_z_neg,
                    cube.XOZ_y_pos,cube.XOZ_y_neg,
                    cube.YOZ_x_pos,cube.YOZ_x_neg);
    //Interracted cube planes. (This number should only be 2, so we abandon the rest.)
    if (numInterPlanes != 2)
        gLevel = 0;
    else
    {
        //find out the two plane indexs
        int planeIdx[2];
        int idx = 0;
        for(int i=0;i<6;i++)
        {
            if(interPlanesBool[i] == 1){
                planeIdx[idx] = i;
                idx++;
            }
        }
        //find out corresponding t1, t2
        double t1 = ts[planeIdx[0]];
        double t2 = ts[planeIdx[1]];
        //make sure that t1 < t2
        if(t1>t2){double temp=t1;t1=t2;t2=temp;}
        //show frame
        if(showFrame)
        {
            double x1 = O->get_x()+t1*D->get_x(); double y1 = O->get_y()+t1*D->get_y(); double z1 = O->get_z()+t1*D->get_z();
            double x2 = O->get_x()+t2*D->get_x(); double y2 = O->get_y()+t2*D->get_y(); double z2 = O->get_z()+t2*D->get_z();
            double xinc = O->get_x()+(tmin+tinc)*D->get_x(), yinc = O->get_y()+(tmin+tinc)*D->get_y(), zinc = O->get_z()+(tmin+tinc)*D->get_z();
            if( ( sqrt(pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2)) <frameWidth) ||
                ( sqrt(pow((x1-xinc),2) + pow((y1-yinc),2) + pow((z1-zinc),2)) <frameWidth/6) ||
                ( sqrt(pow((xinc-x2),2) + pow((yinc-y2),2) + pow((zinc-z2),2)) <frameWidth/6)
                )
            {
                return frameColor;
            }
        }
        //perspection method
        switch(psp)
        {
        case(SURFACE):
        {
            double x1,y1,z1;
            //find out if is sliced
            if(t1>=tmin+tinc)//not sliced
            {
                //calculate coordinate of the nearest intersection point
                x1 = O->get_x()+t1*D->get_x(); y1 = O->get_y()+t1*D->get_y(); z1 = O->get_z()+t1*D->get_z();
            }else{
                //calculate coordinate of the nearest intersection point
                x1 = O->get_x()+(tmin+tinc)*D->get_x(); y1 = O->get_y()+(tmin+tinc)*D->get_y(); z1 = O->get_z()+(tmin+tinc)*D->get_z();
            }
            gLevel = cube.getClosestValue(x1,y1,z1);
            break;
        }
        case(MAX):
        {
            gLevel = 0;
            for(double t=max(tmin+tinc,t1);t<=t2;t=t+(tmax-tmin)*sample_step)
            {
                double x = O->get_x()+t*D->get_x(); double y = O->get_y()+t*D->get_y(); double z = O->get_z()+t*D->get_z();
                if(cube.getClosestValue(x,y,z)>gLevel)
                {
                    gLevel = cube.getClosestValue(x,y,z);
                }
            }
            break;
        }
        case(MEAN):
        {
            double sum = 255;
            int count = 0;
            for(double t=max(tmin+tinc,t1);t<=t2;t=t+(tmax-tmin)*sample_step)
            {
                double x = O->get_x()+t*D->get_x(); double y = O->get_y()+t*D->get_y(); double z = O->get_z()+t*D->get_z();
                sum += cube.getClosestValue(x,y,z);
                count++;
            }
            sum = sum/count;
            gLevel = uchar(sum);
            break;
        }
        case(EMAB):
        {
            //this EMAB is simplified that "absorption_rate" are the same for all volumns
            double C = 0;
            double transmissionRate_accuminate = 1;
            for(double t=max(tmin+tinc,t1);t<=t2;t=t+(tmax-tmin)*sample_step)
            {
                double x = O->get_x()+t*D->get_x(); double y = O->get_y()+t*D->get_y(); double z = O->get_z()+t*D->get_z();
                uchar g = cube.getClosestValue(x,y,z);
                absorption_rate = double(g)/255. * int(g>=transparentThreshold_low) * int(g<=transparentThreshold_high) * transparent_reduce;
                C += double(g) * absorption_rate * transmissionRate_accuminate;
                transmissionRate_accuminate = transmissionRate_accuminate*(1-absorption_rate);
//                if(transmissionRate_accuminate<0.5)
//                    break;
            }
            if(C>255)   C=255;
            gLevel = uchar(C);
            break;
        }
        default:
        {
            //calculate coordinates of the two intersection points
            double x1 = O->get_x()+t1*D->get_x(); double y1 = O->get_y()+t1*D->get_y(); double z1 = O->get_z()+t1*D->get_z();
            double x2 = O->get_x()+t2*D->get_x(); double y2 = O->get_y()+t2*D->get_y(); double z2 = O->get_z()+t2*D->get_z();
            gLevel = cube.getClosestValue(x1,y1,z1);
            break;
        }
        }
    }

    return gLevel;
}
void Widget::singleCurrency(int startj,int endj,double tmin,double tmax,double tinc,
                    enum PERSPECTIVE psp,bool showFrame,double frameWidth,uchar frameColor,
                    double sample_step,double absorption_rate,double transparent_reduce)
{
    for(int i=0;i<canvasHeight;i++)
    {
        for(int j=startj;j<=endj;j++)
        {
            Vec3D D = canvas2Viewport(&cvs,&viewport,i,j,beta);
            //D.print();
            Vec3D cam = viewport.get_cam();
            uchar gLevel = traceRay(&scene,&cam,&D,tmin,tmax,tinc,psp,showFrame,frameWidth,frameColor,
                                    sample_step,absorption_rate,this->transparentThreshold_low,this->transparentThreshold_high,
                                    transparent_reduce);
            cvs.putPixel(i,j,gLevel);
        }
    }
}

void Widget::show3D()
{
    // convert Polar to Cartesian
    double x,y,z;
    y = rho*cos(Pi*beta/180);
    x = rho*sin(Pi*beta/180)*sin(Pi*alpha/180);
    z = rho*sin(Pi*beta/180)*cos(Pi*alpha/180);
    // initialize viewport, with d=1;
    viewport.init(vptWidth,vptHeight,x,y,z,d);
    //concurrency
    int num_threads = thread::hardware_concurrency(); // 获取CPU支持的最大线程数
    int chunk_size = canvasWidth / num_threads; // 将数组分成 num_threads 个块
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? (canvasWidth - 1) : (start + chunk_size - 1);
        threads.push_back(std::thread(&Widget::singleCurrency, this, start, end,tmin,tmax,tinc,
                                 psp,showFrame,frameWidth,frameColor,
                                      sample_step,absorption_rate,this->transparent_reduce)); // 使用引用传递数组
    }
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    //load pixel into qimg
    QImage qImg(cvs.get_width(),cvs.get_height(),QImage::Format_Grayscale8);
    for(int x=0;x<cvs.get_width();x++){
        for(int y=0;y<cvs.get_height();y++){
            uchar a=cvs.getPixel(y,x);
            qImg.setPixel(x,y,qRgb(a,a,a));
        }
    }
    //show
    fitpixmap3D = QPixmap::fromImage(qImg);
    fitpixmap3D = fitpixmap3D.scaled(ui->label_4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_4->setPixmap(fitpixmap3D);
    //show viewport location
    ui->label_alpha->setText(QString("α=")+QString::number(alpha,'f',2));
    ui->label_beta->setText(QString("β=")+QString::number(beta,'f',2));
    ui->label_rho->setText(QString("ρ=")+QString::number(rho,'f',2));
    ui->label_rho_2->setText(QString("ρ=")+QString::number(rho,'f',2));
}
void Widget::on_horizontalSlider_pho_valueChanged(int value)
{
    double rho_new = rhoMin + ((double)value)/100*(rhoMax-rhoMin);
    //ray tracing (need to refresh when rho, d are chaged)
    //refresh tinc, tinc_sensitivity
    double tmin_new = (rho_new+d-sqrt(3)/2)/d;
    double tmax_new = (rho_new+d+sqrt(3)/2)/d;
    tinc = tinc * (tmax_new-tmin_new)/(tmax-tmin);
    tinc_sensitivity = (tmax_new-tmin_new)/100;
    //refresh rho, tmin, tmax
    rho = rho_new;
    tmax = tmax_new;
    tmin = tmin_new;

    show3D();
}

//wheel event
void Widget::wheelEvent(QWheelEvent *event)
{
    QPoint Pos = ui->label_4->mapFrom(this, event->pos());
    double width = ui->label_4->width();
    double height = ui->label_4->height();
    double x = Pos.x();
    double y = Pos.y();
    if( x>=0 && x<width && y>=0 && y<height )
    {
        if(event->angleDelta().y()>0){
            tinc = tinc - tinc_sensitivity;
            if(tinc<0)
                tinc = 0;
        }
        else{
            tinc = tinc + tinc_sensitivity;
            if(tinc>tmax-tmin)
                tinc = tmax-tmin;
        }
        ui->horizontalScrollBar->setValue((int)(tinc/(tmax-tmin)*100));
        show3D();
        event->accept();
    }
}
void Widget::on_horizontalScrollBar_valueChanged(int value)
{
    tinc = (double)(tmax-tmin)/100*value;
    show3D();
}

//mousemove event (drag the trackball)
void Widget::mouseMoveEvent(QMouseEvent *event)
{
    QPoint Pos = ui->label_4->mapFrom(this, event->pos());
    double width = ui->label_4->width();
    double height = ui->label_4->height();
    double x = Pos.x();
    double y = height-1-Pos.y();
    double ratio = min(width,height);
    if (event->buttons() & Qt::LeftButton && x>=0 && x<width && y>=0 && y<height)
    {
        qDebug() << "鼠标拖动，位置：" << Pos;
        //x1,y1,z1, x2,y2,z2
        double x1 = (previous_x-width/2)/ratio;
        double y1 = (previous_y-height/2)/ratio;
        double z1;
        if(x1*x1+y1*y1<=0.125)
            z1 = sqrt(0.25-x1*x1-y1*y1);
        else
            z1 = 0.125/sqrt(x1*x1+y1*y1);
        double x2 = (x-width/2)/ratio;
        double y2 = (y-height/2)/ratio;
        double z2;
        if(x2*x2+y2*y2<=0.125)
            z2 = sqrt(0.25-x2*x2-y2*y2);
        else
            z2 = 0.125/sqrt(x2*x2+y2*y2);
        //alpha1,beta1, alpha2,beta2
        double alpha1 = atan(x1/z1)/Pi*180; double beta1 = atan(y1/sqrt(x1*x1+z1*z1))/Pi*180;
        double alpha2 = atan(x2/z2)/Pi*180; double beta2 = atan(y2/sqrt(x2*x2+z2*z2))/Pi*180;
        qDebug() << "x1: " << x1<< "x2: " << x2;
        qDebug() << "y1: " << y1<< "y2: " << y2;
        qDebug() << "z1: " << z1<< "z2: " << z2;
        qDebug() << "alpha1: " << alpha1<< "alpha2: " << alpha2;
        qDebug() << "beta1: " << beta1<< "beta2: " << beta2;
        qDebug() << "alpha: " << alpha<< "beta: " << beta;
        //update alpha, beta
        if (beta >= 180)
            alpha = alpha - (alpha2-alpha1);
        else
            alpha = alpha + (alpha2-alpha1);
        beta = beta + (beta2-beta1);
        if(alpha < 0)   alpha += 360;
        if(alpha >= 360) alpha -= 360;
        if(beta < 0)   beta += 360;
        if(beta >= 360) beta -= 360;
        //update previous position
        previous_x = x;
        previous_y = y;
        //show3D
        show3D();
    }
    event->accept();
}
//mouse press event
void Widget::mousePressEvent(QMouseEvent * event)
{
    if(event->type() == QEvent::MouseButtonPress)
    {
        if(event->button() == Qt::LeftButton)
        {
            QPoint Pos = ui->label_4->mapFrom(this, event->pos());
            double width = ui->label_4->width();
            double height = ui->label_4->height();
            double x = Pos.x();
            double y = height-1-Pos.y();
            if(x>=0 && x<width && y>=0 && y<height)
            {
                previous_x = x;
                previous_y = y;
                qDebug()<<"";
                qDebug()<<"鼠标左键被按下，相对label_4坐标"<<Pos;
                qDebug() << "label_4大小：" << width << "X" <<height;
            }
        }
    }
    event->accept();
}

void Widget::on_spinBox_valueChanged(int arg1)
{
    pspsen = arg1;
    tinc_sensitivity = (tmax-tmin)/pspsen;
}
//cube side scales
void Widget::on_doubleSpinBox_valueChanged(double arg1)
{
    //2D
    this->xyDisplay_X = int( size_x*arg1 );
    this->xzDisplay_X = int( size_x*arg1 );
    showAll();
    //3D
    cb_scaleX = arg1;
    scene.imagecube.set_scales(cb_scaleX,cb_scaleY,cb_scaleZ);
    scene.imagecube.shift2Origin();
    show3D();
}
void Widget::on_doubleSpinBox_2_valueChanged(double arg1)
{
    //2D
    this->xyDisplay_Y = int( size_y*arg1 );
    this->yzDisplay_Y = int( size_y*arg1 );
    showAll();
    //3D
    cb_scaleY = arg1;
    scene.imagecube.set_scales(cb_scaleX,cb_scaleY,cb_scaleZ);
    scene.imagecube.shift2Origin();
    show3D();
}
void Widget::on_doubleSpinBox_3_valueChanged(double arg1)
{
    //2D
    this->xzDisplay_Y = int( size_z*arg1 );
    this->yzDisplay_X = int( size_z*arg1 );
    showAll();
    //3D
    cb_scaleZ = arg1;
    scene.imagecube.set_scales(cb_scaleX,cb_scaleY,cb_scaleZ);
    scene.imagecube.shift2Origin();
    show3D();
}

//cube frame
void Widget::on_checkBoxShowOriginalSize_2_stateChanged(int arg1)
{
    showFrame=arg1;
    if(showFrame==1){
        ui->doubleSpinBox_4->setEnabled(1);
        show3D();
    }else
    {
        ui->doubleSpinBox_4->setEnabled(0);
        show3D();
    }

}
void Widget::on_doubleSpinBox_4_valueChanged(double arg1)
{
    frameWidth=arg1;
    show3D();
}
//pspc
void Widget::on_comboBox_rayTracing_currentIndexChanged(int index)
{
    switch (index) {
    case 0:
    {
        psp = EMAB;
        ui->doubleSpinBox_5->setEnabled(1);
        break;
    }
    case 1:
    {
        psp = SURFACE;
        ui->doubleSpinBox_5->setEnabled(0);
        break;
    }
    case 2:
    {
        psp = MAX;
        ui->doubleSpinBox_5->setEnabled(1);
        break;
    }
    case 3:
    {
        psp = MEAN;
        ui->doubleSpinBox_5->setEnabled(1);
        break;
    }
    default:
        break;
    }
    show3D();
}
void Widget::on_doubleSpinBox_5_valueChanged(double arg1)
{
    sample_step = arg1;
    show3D();
}

void Widget::resizeEvent(QResizeEvent *event)
{
    if(programmeIsOn)
    {
        showAll();
        show3D();
    }
}

//showLocatingLines
void Widget::on_checkBox_stateChanged(int arg1)
{
    showLocation = arg1;
    show3DSlices = arg1;
    if(arg1)
    {
        ui->checkBox_xOy->setEnabled(true);
        ui->checkBox_xOz->setEnabled(true);
        ui->checkBox_yOz->setEnabled(true);
        scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,this->status,SliceMask);
    }
    else
    {
        ui->checkBox_xOy->setEnabled(false);
        ui->checkBox_xOz->setEnabled(false);
        ui->checkBox_yOz->setEnabled(false);
        scene.imagecube.cancelDrawSlice();
    }
    showAll();
    show3D();
}

//savefiles
void Widget::keyPressEvent(QKeyEvent *event)
{
    if (event->modifiers() == Qt::CTRL && event->key() == Qt::Key_S)
    {
        QStringList filePathList = this->filePath.split("/");
        QString filename = QFileDialog::getSaveFileName(this,"请选择保存路径",filePathList[filePathList.length()-1],tr("*.bmp;; *.png;; *.jpg;; *.tif;; *.GIF"));
        if(filename.isEmpty())
            return;
        QString f1 = filename.split('.')[0];
        QString f2 = filename.split('.')[1];
        fitpixmapXOY.save(f1 + QString("_XOY.") + f2);
        fitpixmapXOZ.save(f1 + QString("_XOZ.") + f2);
        fitpixmapYOZ.save(f1 + QString("_YOZ.") + f2);
        fitpixmap3D.save(f1 + QString("_rendering.") + f2);
        QMessageBox::information(this,"提示","保存成功！");
    }
}

//transparent thresholds
void Widget::on_lowThreshold_valueChanged(int arg1)
{
    if(arg1<=transparentThreshold_high)
    {
        this->transparentThreshold_low = arg1;
        show3D();
        showAll();
    }
}
void Widget::on_highThreshold_valueChanged(int arg1)
{
    if(arg1>=transparentThreshold_low)
    {
        this->transparentThreshold_high = arg1;
        show3D();
        showAll();
    }
}
void Widget::on_highThreshold_editingFinished()
{
    int arg1 = ui->highThreshold->value();
    if(arg1<=transparentThreshold_low)
        ui->highThreshold->setValue(transparentThreshold_low);
    this->transparentThreshold_high = arg1;
    show3D();
    showAll();
}
void Widget::on_lowThreshold_editingFinished()
{
    int arg1 = ui->lowThreshold->value();
    if(arg1>=transparentThreshold_high)
        ui->lowThreshold->setValue(transparentThreshold_high);
    this->transparentThreshold_low = arg1;
    show3D();
    showAll();
}

//3D slice show
void Widget::on_checkBox_xOy_stateChanged(int arg1)
{
    this->status[0] = arg1;
    scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,this->status,SliceMask);
    show3D();
}
void Widget::on_checkBox_xOz_stateChanged(int arg1)
{
    this->status[1] = arg1;
    scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,this->status,SliceMask);
    show3D();
}
void Widget::on_checkBox_yOz_stateChanged(int arg1)
{
    this->status[2] = arg1;
    scene.imagecube.drawSlice(xyPlaneIdx,xzPlaneIdx,yzPlaneIdx,this->status,SliceMask);
    show3D();
}


void Widget::on_horizontalSlider_valueChanged(int value)
{
    transparent_reduce = 1-(double)value/100.;
    show3D();
}
