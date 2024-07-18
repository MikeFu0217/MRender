#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QLabel>
#include "array3D.h"
#include "array2D.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

enum interpolationMethods {Nearest, Bilinear};
enum PERSPECTIVE {EMAB,SURFACE,MAX,MEAN};

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    void chooseFile();
    void loadVideo2Array3D();
    void refresh();
    void xSliceChanged(int x);
    void ySliceChanged(int y);
    void zSliceChanged(int z);
    void showAll();
    void labelShow(QLabel *lb, QPixmap *pxm);
    void xyShow(int z);
    void xzShow(int y);
    void yzShow(int x);
    //3D display
    void show3D();
    void singleCurrency(int startj,int endj,double tmin,double tmax,double tinc,
                        enum PERSPECTIVE psp = SURFACE,bool showFrame=true,double frameWidth=0.005,uchar frameColor=255,
                        double sample_step=0.01,double absorption_rate=0.5,double transparent_reduce=1);

private slots:
    void on_loadFileButton_clicked();

    void on_spinBox_1_valueChanged(int arg1);

    void on_spinBox_2_valueChanged(int arg1);

    void on_spinBox_3_valueChanged(int arg1);

    void on_comboBox_SR_currentIndexChanged(int index);

    void on_spinBox_valueChanged(int arg1);

    void on_doubleSpinBox_valueChanged(double arg1);

    void on_doubleSpinBox_2_valueChanged(double arg1);

    void on_doubleSpinBox_3_valueChanged(double arg1);

    void on_checkBoxShowOriginalSize_2_stateChanged(int arg1);

    void on_doubleSpinBox_4_valueChanged(double arg1);

    void on_comboBox_rayTracing_currentIndexChanged(int index);

    void on_doubleSpinBox_5_valueChanged(double arg1);

    void on_checkBox_stateChanged(int arg1);

    void on_lowThreshold_valueChanged(int arg1);

    void on_highThreshold_valueChanged(int arg1);

    void on_highThreshold_editingFinished();

    void on_lowThreshold_editingFinished();

    void on_checkBox_xOy_stateChanged(int arg1);

    void on_checkBox_xOz_stateChanged(int arg1);

    void on_checkBox_yOz_stateChanged(int arg1);

    void on_horizontalScrollBar_X_valueChanged(int value);

    void on_horizontalScrollBar_Y_valueChanged(int value);

    void on_horizontalScrollBar_Z_valueChanged(int value);

    void on_horizontalSlider_pho_valueChanged(int value);

    void on_horizontalScrollBar_valueChanged(int value);

    void on_horizontalSlider_valueChanged(int value);

    void on_checkBox_2_stateChanged(int arg1);

private:
    Ui::Widget *ui;
    QStringList imagePaths;
    QString filePath;
    // img3D
    int size_x=1, size_y=1, size_z=1;
    Array3D<uchar> img3D;
    // xy,xz,yz planes
    Array2D<uchar> xyPlane, xzPlane, yzPlane;
    // parameters for locating lines
    int xyPlaneIdx=0, xzPlaneIdx=0, yzPlaneIdx=0;
    int showLocation = 1;
    int showLocationWIdth = 1;
    uchar showLocationColor = 255;
    // parameters for Display
    int xyDisplay_X=300,xyDisplay_Y=300,xzDisplay_X=300,xzDisplay_Y=300,yzDisplay_X=300,yzDisplay_Y=300;
    int xyDisplay_X_Cache=300,xyDisplay_Y_Cache=300,xzDisplay_X_Cache=300,xzDisplay_Y_Cache=300,yzDisplay_X_Cache=300,yzDisplay_Y_Cache=300;
    void setDisplaysEnabled(bool judge);
    enum interpolationMethods interpolationMethod=Nearest;
    //Scene
    double cb_scaleX=1,cb_scaleY=1,cb_scaleZ=1;
    //3D display
    double alpha = 135;
    double beta = 240;
    double rhoMin=0.5, rhoMax=4;
    double rho = 2;
    double d=1;
    int transparentThreshold_low = 0;
    int transparentThreshold_high = 255;
    //ray tracing (need to refresh when rho, d are chaged)
    double tmin = (rho+d-sqrt(3)/2)/d;
    double tmax = (rho+d+sqrt(3)/2)/d;
    int pspsen = 50;
    double tinc = 0, tinc_sensitivity = (tmax-tmin)/pspsen;
    enum PERSPECTIVE psp = EMAB;
    double sample_step = 0.005;
    double absorption_rate = 0.1;
    double transparent_reduce = 1;
    //frame
    bool showFrame=true;
    double frameWidth=0.02;
    uchar frameColor=255;
    //mouse events
    double previous_x=0, previous_y=0;
    //resizeEvent
    bool programmeIsOn = false;
    //reload file
    bool allowShowAll = true;
    //QPixmap
    QPixmap fitpixmapXOY,fitpixmapXOZ,fitpixmapYOZ,fitpixmap3D;
    //show 3D slices
    bool show3DSlices = true;
    uchar SliceMask = 150;
    bool status[3] = {0,0,0};
    //use GPU
    int GPUisON = 1;

protected:
    void wheelEvent(QWheelEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent *event);
    void keyPressEvent(QKeyEvent *event);
};
#endif // WIDGET_H
