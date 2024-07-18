#ifndef SCENE_H
#define SCENE_H

#include "imagecube.h"
#include "array3D.h"

class Scene
{
public:
    ImageCube imagecube;

    Scene(){};
    void addCube(Array3D<uchar> *array3D,double xscale,double yscale,double zscale,bool toOrigin=true)
    {
        imagecube.init(array3D,xscale,yscale,zscale);
        if(toOrigin)
            imagecube.shift2Origin();
    }
    ImageCube getCube(){return imagecube;}
};

#endif // SCENE_H
