#ifndef ARRAY3D_H
#define ARRAY3D_H

template<class T>
class Array3D
{
    int size_x=1, size_y=1, size_z=1;
    T* head = nullptr;
    bool isAllocated = 0;
public:
    Array3D();
    Array3D(int sz_x, int sz_y, int sz_z);
    ~Array3D();

    void malloc(int sz_x, int sz_y, int sz_z);
    void destroy();
    T at(int x, int y, int z);
    void set(int x, int y, int z,T value);
    int get_size_x(){return size_x;}
    int get_size_y(){return size_y;}
    int get_size_z(){return size_z;}
};

template<class T>
Array3D<T>::Array3D()
{}
template<class T>
Array3D<T>::Array3D(int sz_x, int sz_y, int sz_z):size_x(sz_x),size_y(sz_y),size_z(sz_z)
{
    head = new T[sz_x*sz_y*sz_z];
    isAllocated = 1;
}
template<class T>
Array3D<T>::~Array3D()
{
    delete[] head;
    isAllocated = 0;
}

template<class T>
void Array3D<T>::malloc(int sz_x, int sz_y, int sz_z)
{
    if(!isAllocated)
    {
        this->size_x = sz_x; this->size_y = sz_y; this->size_z = sz_z;
        head = new T[sz_x*sz_y*sz_z];
        isAllocated = 1;
    }
}

template<class T>
void Array3D<T>::destroy()
{
    if(isAllocated)
    {
        delete[] head;
        head = nullptr;
        isAllocated = 0;
    }
}

template<class T>
T Array3D<T>::at(int x, int y, int z)
{
    return head[ x*(this->size_y*this->size_z)+(y*this->size_z+z) ];
}
template<class T>
void Array3D<T>::set(int x,int y,int z,T value)
{
    head[ x*(this->size_y*this->size_z)+(y*this->size_z+z) ] = value;
}

#endif // ARRAY3D_H
