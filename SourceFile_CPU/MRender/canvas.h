#ifndef CANVAS_H
#define CANVAS_H

typedef unsigned char uchar;

class Canvas
{
    int width, height;
    uchar* data;
    int size;
public:
    Canvas(int w, int h, uchar init);
    ~Canvas();
    int get_width(){return width;}
    int get_height(){return height;}
    int get_size(){return size;}
    uchar at(int i);
    void set(int i,uchar value);
    void putPixel(int i,int j,uchar value);
    uchar getPixel(int i,int j);
};

Canvas::Canvas(int w, int h, uchar initValue = 0):width(w),height(h),size(w*h)
{
    data = new uchar[size];

    for(int i=0;i<size;i++)
        data[i] = initValue;
}
Canvas::~Canvas()
{
    delete[] data;
}
uchar Canvas::at(int i)
{
    return data[i];
}
void Canvas::set(int i,uchar value)
{
    data[i] = value;
}
void Canvas::putPixel(int i,int j,uchar value)
{
    data[i*width+j] = value;
}
uchar Canvas::getPixel(int i,int j)
{
    return data[i*width+j];
}
#endif // CANVAS_H
