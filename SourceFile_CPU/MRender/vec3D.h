#ifndef VEC3D_H
#define VEC3D_H

#include <math.h>
#include <iostream>

using namespace std;

class Vec3D
{
    double x=0,y=0,z=0;
public:
    Vec3D(){}
    Vec3D(double xx,double yy,double zz):x(xx),y(yy),z(zz){}
    Vec3D(double x1,double y1,double z1,double x2,double y2,double z2):x(x1-x2),y(y1-y2),z(z1-z2){}
    Vec3D(Vec3D *vec1,Vec3D *vec2):x(vec1->get_x()-vec2->get_x()),y(vec1->get_y()-vec2->get_y()),z(vec1->get_z()-vec2->get_z()){}
    void init(double xx,double yy,double zz){x=xx,y=yy,z=zz;}
    double get_x(){return x;}
    double get_y(){return y;}
    double get_z(){return z;}
    void print(){cout<<"("<<x<<","<<y<<","<<z<<")"<<endl;}
    static double dot(Vec3D *X, Vec3D *Y);
    static Vec3D add(Vec3D *X, Vec3D *Y);
    static Vec3D numMul(double a,Vec3D *X);
    Vec3D unit();
    double get_2norm();
};

double Vec3D::dot(Vec3D *X,Vec3D *Y)
{
    return X->get_x()*Y->get_x()+X->get_y()*Y->get_y()+X->get_z()*Y->get_z();
}
Vec3D Vec3D::add(Vec3D *X,Vec3D *Y)
{
    Vec3D result(X->get_x()+Y->get_x(),X->get_y()+Y->get_y(),X->get_z()+Y->get_z());
    return result;
}
Vec3D Vec3D::unit()
{
    double scaler = pow(pow(x,2)+pow(y,2)+pow(z,2),0.5);
    Vec3D result(x/scaler,y/scaler,z/scaler);
    return result;
}
Vec3D Vec3D::numMul(double t,Vec3D *X)
{
    Vec3D result(X->get_x()*t,X->get_y()*t,X->get_z()*t);
    return result;
}
double Vec3D::get_2norm()
{

    return sqrt(x*x+y*y+z*z);
}

#endif // VEC3D_H
