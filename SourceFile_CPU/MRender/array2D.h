#ifndef ARRAY2D_H
#define ARRAY2D_H

#include <QDebug>
#include "array3D.h"
#include <math.h>

enum planes {XY, XZ, YZ};

template<class T>
class Array2D
{
    int size_x=1, size_y=1;
    T* head = nullptr;
    bool isAllocated = 0;
public:
    Array2D();
    Array2D(int sz_x, int sz_y);
    ~Array2D();

    void malloc(int sz_x, int sz_y);
    void destroy();
    T at(int x, int y);
    void set(int x, int y, T value);
    void loadFromArray3D(Array3D<T> *ptr_array3D, enum planes plane, int i);
    T* data(){return head;}
    void drawLocatingLines(int xloc, int yloc, int width=1, unsigned char grayLevel=255);
    // interpolation
    void nearest(Array2D<T> *imgAfter);
    void bilinear(Array2D<T> *imgAfter);
    int get_size_x(){return size_x;}
    int get_size_y(){return size_y;}
};

template<class T>
Array2D<T>::Array2D()
{}
template<class T>
Array2D<T>::Array2D(int sz_x, int sz_y):size_x(sz_x),size_y(sz_y)
{
    head = new T[sz_x*sz_y];
    isAllocated = 1;
}
template<class T>
Array2D<T>::~Array2D<T>()
{
    delete[] head;
    isAllocated = 0;
}

template<class T>
void Array2D<T>::malloc(int sz_x, int sz_y)
{
    if(!isAllocated)
    {
        this->size_x=sz_x; this->size_y=sz_y;
        head = new T[sz_x*sz_y];
        isAllocated = 1;
    }
}
template<class T>
void Array2D<T>::destroy()
{
    if(isAllocated)
    {
        delete[] head;
        isAllocated = 0;
        head = nullptr;
    }
}

template<class T>
T Array2D<T>::at(int x, int y)
{
    return head[x*this->size_y+y];
}
template<class T>
void Array2D<T>::set(int x, int y, T value)
{

    head[x*this->size_y+y] = value;
}
template<class T>
void Array2D<T>::loadFromArray3D(Array3D<T> *ptr_array3D, enum planes plane, int i)
{
    switch (plane) {
    case XY:
        if(!(this->size_x==ptr_array3D->get_size_x() && this->size_y==ptr_array3D->get_size_y()))
        {
            qDebug()<<"array2D and 3D plane size do not match"<<Qt::endl;
            break;
        }
        for(int x=0;x<this->size_x;x++)
        {
            for(int y=0;y<this->size_y;y++)
            {
                this->set(x,y,ptr_array3D->at(x, y, i));
            }
        }
        break;
    case XZ:
        if(!(this->size_x==ptr_array3D->get_size_x() && this->size_y==ptr_array3D->get_size_z()))
        {
            qDebug()<<"array2D and 3D plane size do not match"<<Qt::endl;
            break;
        }
        for(int x=0;x<this->size_x;x++)
        {
            for(int y=0;y<this->size_y;y++)
            {
                this->set(x,y,ptr_array3D->at(x, i, y));
            }
        }
        break;
    case YZ:
        if(!(this->size_x==ptr_array3D->get_size_z() && this->size_y==ptr_array3D->get_size_y()))
        {
            qDebug()<<"array2D and 3D plane size do not match"<<Qt::endl;
            break;
        }
        for(int x=0;x<this->size_x;x++)
        {
            for(int y=0;y<this->size_y;y++)
            {
                this->set(x,y,ptr_array3D->at(i, y, x));
            }
        }
        break;
    default:
        qDebug()<<"Wrong input parameter."<<Qt::endl;
    }
}
template<class T>
void Array2D<T>::drawLocatingLines(int xloc, int yloc, int width, unsigned char grayLevel)
{
    for(int x=xloc-width/2;x<=xloc+width/2;x++)
    {
        if(x<0 || x>=this->size_x)
            continue;
        for(int y=0;y<this->size_y;y++)
        {
            this->set(x, y, grayLevel);
        }
    }
    for(int y=yloc-width/2;y<=yloc+width/2;y++)
    {
        if(y<0 || y>=this->size_y)
            continue;
        for(int x=0;x<this->size_x;x++)
        {
            this->set(x, y, grayLevel);
        }
    }
}
template<class T>
void Array2D<T>::nearest(Array2D<T> *imgAfter)
{
    int wBefore = this->size_x, hBefore = this->size_y;
    int wAfter = imgAfter->get_size_x(), hAfter = imgAfter->get_size_y();
    for(int xAfter=0;xAfter<wAfter;xAfter++)
    {
        for(int yAfter=0;yAfter<hAfter;yAfter++)
        {
            int xBefore = round( (double)xAfter * (double)(wBefore-1) / (double)(wAfter-1) );
            int yBefore = round( (double)yAfter * (double)(hBefore-1) / (double)(hAfter-1) );
            if(xBefore>=size_x || yBefore>=size_y)
                qDebug()<<"xBefore="<<xBefore<<", yBefore="<<yBefore<<Qt::endl;
            imgAfter->set(xAfter, yAfter, this->at(xBefore, yBefore));
        }
    }
}
template<class T>
void Array2D<T>::bilinear(Array2D<T> *imgAfter)
{
    for(int i=0;i<imgAfter->get_size_x();i++)
    {
        for(int j=0;j<imgAfter->get_size_y();j++)
        {
            double x = (double)i * (double)(this->size_x-1) / (double)(imgAfter->get_size_x()-1);
            double y = (double)j * (double)(this->size_y-1) / (double)(imgAfter->get_size_y()-1);
            uchar p=0;
            int x1 = floor(x), y1 = floor(y);
            int x2 = ceil(x), y2 = ceil(y);
            if(x==(int)x && y==(int)y)
            {
                p = this->at(x,y);
            }
            else if(x==(int)x && y!=(int)y)
            {
                double beta = (y-(double)y1)/((double)y2-(double)y1);
                p = (uchar)round(beta*(double)this->at(x,y1)+
                                 (1-beta)*(double)this->at(x,y2));
            }
            else if(x!=(int)x && y==(int)y)
            {
                double alpha = (x-(double)x1)/((double)x2-(double)x1);
                p = (uchar)round(alpha*(double)this->at(x1,y)+
                                  (1-alpha)*(double)this->at(x2,y));
            }
            else
            {
                double alpha = (x-(double)x1)/((double)x2-(double)x1);
                double beta = (y-(double)y1)/((double)y2-(double)y1);
                p = (uchar)round(alpha*beta*(double)this->at(x1,y1)+
                              (1-alpha)*beta*(double)this->at(x2,y1)+
                              alpha*(1-beta)*(double)this->at(x1,y2)+
                              (1-alpha)*(1-beta)*(double)this->at(x2,y2));
            }
            imgAfter->set(i, j, p);
        }
    }
}

#endif // ARRAY2D_H
