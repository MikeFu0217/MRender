#ifndef IMAGECUBE_H
#define IMAGECUBE_H

#include "array3D.h"
#include "vec3D.h"
#include <math.h>
#include <QDebug>
typedef unsigned char uchar;

double myMax(double a, double b, double c)
{
    if(a>b)
    {
        if(a>c)
            return a;
        else
            return c;
    }else{
        if(b>c)
            return b;
        else
            return c;
    }
}

class ImageCube
{
    //parameters concerning array3D data
    Array3D<uchar> *array3D = nullptr;
    double xScale,yScale,zScale; //scale for scaling in different directions
    //parameters for imagecube
    double xlength,ylength,zlength,maxlength; //length of the cube sides in the scene
    double xshift=0,yshift=0,zshift=0;
    Vec3D center; //center of the cube
    //draw slices
    bool drawSliceBool = true;
    int idx,idy,idz;
    uchar mask=100;
    bool status[3];
public:
    ImageCube(){}
    ImageCube(Array3D<uchar> *a, double xsc, double ysc, double zsc);
    void init(Array3D<uchar> *a, double xsc, double ysc, double zsc);
    void set_scales(double xsc, double ysc, double zsc);
    void shift(double xshift, double yshift, double zshift);
    void shift2Origin();
    uchar getClosestValue(double x, double y, double z);
    //get the center and boundary of the cube
    Vec3D get_center(){return center;}
    double get_xlength(){return xlength;}
    double get_ylength(){return ylength;}
    double get_zlength(){return zlength;}
    //6 cube sides locations
    double XOY_z_pos;
    double XOY_z_neg;
    double XOZ_y_pos;
    double XOZ_y_neg;
    double YOZ_x_pos;
    double YOZ_x_neg;
    //draw slices
    void drawSlice(int xyPlaneIdx, int xzPlaneIdx, int yzPlaneIdx, bool *status, uchar mask);
    void cancelDrawSlice();
};

ImageCube::ImageCube(Array3D<uchar> *a, double xsc, double ysc, double zsc):array3D(a),xScale(xsc),yScale(ysc),zScale(zsc)
{
    xlength=xScale*(array3D->get_size_x()-1); ylength=yScale*(array3D->get_size_y()-1); zlength=zScale*(array3D->get_size_z()-1);
    //do normalization to make the lagest side-length to be 1
    maxlength = myMax(xlength, ylength, zlength);
    xlength=xlength/maxlength; ylength=ylength/maxlength; zlength=zlength/maxlength;
    //initialize center
    center.init(xlength/2,ylength/2,zlength/2);
    //6 cube sides
    XOY_z_pos = center.get_z()+zlength/2;
    XOY_z_neg = center.get_z()-zlength/2;
    XOZ_y_pos = center.get_y()+ylength/2;
    XOZ_y_neg = center.get_y()-ylength/2;
    YOZ_x_pos = center.get_x()+xlength/2;
    YOZ_x_neg = center.get_x()-xlength/2;
};

void ImageCube::init(Array3D<uchar> *a, double xsc, double ysc, double zsc)
{
    array3D = a;
    xScale = xsc; yScale = ysc; zScale = zsc;
    xlength=xScale*(array3D->get_size_x()-1); ylength=yScale*(array3D->get_size_y()-1); zlength=zScale*(array3D->get_size_z()-1);
    //do normalization to make the lagest side-length to be 1
    maxlength = myMax(xlength, ylength, zlength);
    xlength=xlength/maxlength; ylength=ylength/maxlength; zlength=zlength/maxlength;
    //initialize center
    center.init(xlength/2,ylength/2,zlength/2);
    //6 cube sides
    XOY_z_pos = center.get_z()+zlength/2;
    XOY_z_neg = center.get_z()-zlength/2;
    XOZ_y_pos = center.get_y()+ylength/2;
    XOZ_y_neg = center.get_y()-ylength/2;
    YOZ_x_pos = center.get_x()+xlength/2;
    YOZ_x_neg = center.get_x()-xlength/2;
}
void ImageCube::set_scales(double xsc, double ysc, double zsc)
{
    xScale = xsc; yScale = ysc; zScale = zsc;
    xlength=xScale*(array3D->get_size_x()-1); ylength=yScale*(array3D->get_size_y()-1); zlength=zScale*(array3D->get_size_z()-1);
    //do normalization to make the lagest side-length to be 1
    maxlength = myMax(xlength, ylength, zlength);
    xlength=xlength/maxlength; ylength=ylength/maxlength; zlength=zlength/maxlength;
    //initialize center
    center.init(xlength/2,ylength/2,zlength/2);
    //6 cube sides
    XOY_z_pos = center.get_z()+zlength/2;
    XOY_z_neg = center.get_z()-zlength/2;
    XOZ_y_pos = center.get_y()+ylength/2;
    XOZ_y_neg = center.get_y()-ylength/2;
    YOZ_x_pos = center.get_x()+xlength/2;
    YOZ_x_neg = center.get_x()-xlength/2;
}

void ImageCube::shift(double xshift, double yshift, double zshift)
{
    this->xshift=xshift; this->yshift=yshift; this->zshift=zshift;
    double x = center.get_x(); double y = center.get_y(); double z = center.get_z();
    center.init(x-xshift,y-yshift,z-zshift);
    //6 cube sides
    XOY_z_pos = center.get_z()+zlength/2;
    XOY_z_neg = center.get_z()-zlength/2;
    XOZ_y_pos = center.get_y()+ylength/2;
    XOZ_y_neg = center.get_y()-ylength/2;
    YOZ_x_pos = center.get_x()+xlength/2;
    YOZ_x_neg = center.get_x()-xlength/2;
}

void ImageCube::shift2Origin()
{
    shift(xlength/2, ylength/2, zlength/2);
}

uchar ImageCube::getClosestValue(double x, double y, double z)
{
    int xind=round((x+xshift)*maxlength/xScale); int yind=round((y+yshift)*maxlength/yScale); int zind=round((z+zshift)*maxlength/zScale); //shift back, scale back
    if(xind<0 || xind>=array3D->get_size_x() || yind<0 || yind>=array3D->get_size_y() || zind<0 || zind>=array3D->get_size_z())
    {
        //qDebug()<<"cube index out of range";
        return 0;
    }
    if(!drawSliceBool)
        return array3D->at(xind, yind, zind);
    //draw 3D slices
    if(drawSliceBool)
    {
        if( status[0]==1 && abs( x-(idx*xScale/maxlength-xshift) ) <= 0.005 )
            return min(array3D->at(xind, yind, zind)+mask,255);
        if( status[1]==1 && abs( y-(idy*yScale/maxlength-yshift) ) <= 0.005 )
            return min(array3D->at(xind, yind, zind)+mask,255);
        if( status[2]==1 && abs( z-(idz*zScale/maxlength-zshift) ) <= 0.005 )
            return min(array3D->at(xind, yind, zind)+mask,255);
    }
    return array3D->at(xind, yind, zind);
}

void ImageCube::drawSlice(int xyPlaneIdx, int xzPlaneIdx, int yzPlaneIdx, bool *status, uchar mask=255)
{
    drawSliceBool = true;
    idx = yzPlaneIdx; idy = xzPlaneIdx; idz = xyPlaneIdx;
    this->mask = mask;
    for(int i=0;i<3;i++)
        this->status[i] = status[i];
}

void ImageCube::cancelDrawSlice()
{
    drawSliceBool = false;
}

#endif // IMAGECUBE_H
