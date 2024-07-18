#ifndef VIEWPORT_H
#define VIEWPORT_H
#include "vec3D.h"

class Viewport
{
public:
    double width=1, height=1, d=1;
    Vec3D center;
    Vec3D camera;
    Viewport(){}
    Viewport(double w,double h,Vec3D cen,double dd);
    Viewport(double w,double h,double ceX,double ceY,double ceZ,double dd);
    void init(double w,double h,double ceX,double ceY,double ceZ,double dd);
    double get_width(){return width;}
    double get_height(){return height;}
    Vec3D get_cen(){return center;}
    Vec3D get_cam(){return camera;}
    Vec3D ca_cen();
    Vec3D cen_v(double x,double y,double beta);
};
Viewport::Viewport(double w,double h,Vec3D cen,double dd):width(w),height(h),d(dd)
{
    center = cen;
    double alpha = dd/center.get_2norm() + 1;
    camera = Vec3D::numMul(alpha, &center);
}
Viewport::Viewport(double w,double h,double ceX,double ceY,double ceZ,double dd):width(w),height(h),d(dd)
{
    center.init(ceX,ceY,ceZ);
    double alpha = dd/center.get_2norm() + 1;
    camera = Vec3D::numMul(alpha, &center);
}
void Viewport::init(double w,double h,double ceX,double ceY,double ceZ,double dd)
{
    width = w;
    height = h;
    d = dd;
    center.init(ceX,ceY,ceZ);
    double alpha = dd/center.get_2norm() + 1;
    camera = Vec3D::numMul(alpha, &center);
}
//vector from camera to center
Vec3D Viewport::ca_cen()
{
    Vec3D result(&center,&camera);
    return result;
}
Vec3D Viewport::cen_v(double x,double y,double beta)
{
    // calculate vector from center to V
    double xc=center.get_x(), yc=center.get_y(), zc=center.get_z();
    double SQR = sqrt(pow(zc,2)+pow(xc,2));
    double Cunit = sqrt(pow(zc,2)+pow(yc,2)+pow(xc,2));
    Vec3D m,n;
    if(beta<=180){
        m.init(-1*zc/SQR, 0, xc/SQR);
        n.init(-1*xc*yc/Cunit/SQR, SQR/Cunit, -1*yc*zc/Cunit/SQR);
    }else{
        m.init(zc/SQR, 0, -1*xc/SQR);
        n.init(xc*yc/Cunit/SQR, -1*SQR/Cunit, yc*zc/Cunit/SQR);
    }
    Vec3D CVx = Vec3D::numMul(x,&m);
    Vec3D CVy = Vec3D::numMul(y,&n);
    Vec3D result = Vec3D::add( &CVx, &CVy );
    return result;
}
#endif // VIEWPORT_H
