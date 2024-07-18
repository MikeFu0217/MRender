#include "rayTrace_GPU.h"
#include <cuda_runtime.h>
#include "stdio.h"
#include <math.h>

__device__ uchar at(uchar *cb_array3D_head, int x, int y, int z, int size_x, int size_y, int size_z)
{
    //printf("at(%d,%d,%d) from (%d,%d,%d)\n",x,y,z,size_x,size_y,size_z);
    //printf("at(%d,%d,%d)\n",x,y,z);
    //printf("from (%d,%d,%d)\n",size_x,size_y,size_z);
    //return 50;
    return cb_array3D_head[ x*(size_y*size_z)+(y*size_z+z) ];
}

__device__ uchar getClosestValue(uchar *cb_array3D_head,double x, double y, double z,
                                 double cb_xScale, double cb_yScale, double cb_zScale,
                                 double cb_maxlength,
                                 double cb_xshift, double cb_yshift, double cb_zshift,
                                 int cb_array3D_sizex, int cb_array3D_sizey, int cb_array3D_sizez,
                                 bool cb_drawSliceBool, uchar cb_mask, bool *cb_status,
                                 int idx, int idy, int idz)
{
    int xind=round((x+cb_xshift)*cb_maxlength/cb_xScale);
    int yind=round((y+cb_yshift)*cb_maxlength/cb_yScale);
    int zind=round((z+cb_zshift)*cb_maxlength/cb_zScale); //shift back, scale back
    //printf("at(%d,%d,%d)\n",xind,yind,zind);
    if(xind<0 || xind>=cb_array3D_sizex || yind<0 || yind>=cb_array3D_sizey || zind<0 || zind>=cb_array3D_sizez)
    {
        //qDebug()<<"cube index out of range";
        return 0;
    }
    //printf("xyzindex=(%d,%d,%d) array3Dsize=(%d,%d,%d)\n",xind,yind,zind,cb_array3D_sizex,cb_array3D_sizey,cb_array3D_sizez);
    //printf("cb_status:(%d,%d,%d)\n",cb_status[0],cb_status[1],cb_status[2]);
    if(!cb_drawSliceBool)
        return at(cb_array3D_head, xind, yind, zind, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez);
    //draw 3D slices
    if(cb_drawSliceBool)
    {
        if( cb_status[0]==1 && abs( x-(idx*cb_xScale/cb_maxlength-cb_xshift) ) <= 0.005 )
            return min(at(cb_array3D_head, xind, yind, zind, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez)+cb_mask,255);
        if( cb_status[1]==1 && abs( y-(idy*cb_yScale/cb_maxlength-cb_yshift) ) <= 0.005 )
            return min(at(cb_array3D_head, xind, yind, zind, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez)+cb_mask,255);
        if( cb_status[2]==1 && abs( z-(idz*cb_zScale/cb_maxlength-cb_zshift) ) <= 0.005 )
            return min(at(cb_array3D_head, xind, yind, zind, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez)+cb_mask,255);
    }
    uchar a = at(cb_array3D_head, xind, yind, zind, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez);
    return a;
}

__device__ void set_cvs_data_GPU(uchar *cvs_data_GPU, uchar value,
                                 int cvs_width, int i, int j)
{
    cvs_data_GPU[i*cvs_width + j] = value;
}

__global__ void traceSinglePixel(uchar *cvs_data_GPU,
                                 int cvs_height, int cvs_width,
                                 double vpt_witdh, double vpt_height,
                                 double xc, double yc, double zc,double beta,
                                 double xca, double yca, double zca,
                                 //intercubes
                                 double cb_xScale, double cb_yScale, double cb_zScale,
                                 double cb_xlength, double cb_ylength, double cb_zlength, double cb_maxlength,
                                 double cb_xshift, double cb_yshift, double cb_zshift,
                                 double cb_centerx, double cb_centery, double cb_centerz,
                                 double XOY_z_pos, double XOY_z_neg,
                                 double XOZ_y_pos, double XOZ_y_neg,
                                 double YOZ_x_pos, double YOZ_x_neg,
                                 bool cb_drawSliceBool,
                                 int cb_idx, int cb_idy, int cb_idz,
                                 uchar cb_mask, bool *cb_status,
                                 uchar *cb_array3D_head,int cb_array3D_sizex, int cb_array3D_sizey, int cb_array3D_sizez,
                                 //traceray
                                 double tmin,double tmax,double tinc,
                                 int psp,
                                 bool showFrame,double frameWidth,uchar frameColor,
                                 double sample_step,double absorption_rate,double transparent_reduce,
                                 double transparentThreshold_low, double transparentThreshold_high)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int N = cvs_height*cvs_width;
    if(id>=N)   return;

    //find out i and j on the canvas
    int i = id/cvs_width;
    int j = id%cvs_width;

    //canvas2Viewport (get x and y of V in the scene)
    double x = j * vpt_witdh / (cvs_width-1);
    x = x - vpt_witdh/2;
    double y = i * vpt_height / (cvs_height-1);
    y = -1*y + vpt_height/2;
    //canvas2Viewport (cenV)
    double cenVx, cenVy, cenVz;//output
    double SQR = sqrt(pow(zc,2)+pow(xc,2));
    double Cunit = sqrt(pow(zc,2)+pow(yc,2)+pow(xc,2));
    double mx,my,mz,nx,ny,nz;
    if(beta<=180){
        mx=-1*zc/SQR, my=0, mz=xc/SQR;
        nx=-1*xc*yc/Cunit/SQR, ny=SQR/Cunit, nz=-1*yc*zc/Cunit/SQR;
    }else{
        mx=zc/SQR, my=0, mz=-1*xc/SQR;
        nx=xc*yc/Cunit/SQR, ny=-1*SQR/Cunit, nz=yc*zc/Cunit/SQR;
    }
    mx=x*mx, my=x*my, mz=x*mz;
    nx=y*nx, ny=y*ny, nz=y*nz;
    cenVx=mx+nx, cenVy=my+ny, cenVz=mz+nz;
    //canvas2Viewport (ca_cen)
    double caCenx, caCeny, caCenz;//output
    caCenx=xc-xca, caCeny=yc-yca, caCenz=zc-zca;
    //canvas2Viewport (D)
    double Dx, Dy, Dz;//output
    Dx=caCenx+cenVx, Dy=caCeny+cenVy, Dz=caCenz+cenVz;
    //printf("ID(%d): pixel(%d,%d) D=(%.2f,%.2f,%.2f) camera=(%.2f,%.2f,%.2f)\n",id,i,j,Dx,Dy,Dz,xca,yca,zca);

    //traceRay
    uchar gLevel = 0;
    //set_cvs_data_GPU(cvs_data_GPU, gLevel, cvs_width, i, j);
    int numInterPlanes = 0;
    int interPlanesBool[6] = {0,0,0,0,0,0};
    double ts[6] = {0,0,0,0,0,0};
    //interCubePlanes (start)
    double t,z;
    if(Dx==0 && Dy==0)//XOY_pos,XOY_neg
    {
        numInterPlanes=2;
        interPlanesBool[0]=1; interPlanesBool[1]=1;
        ts[0]=(XOY_z_pos - zca) / Dz;
        ts[1]=(XOY_z_neg - zca) / Dz;
    }
    else if(Dx==0 && Dz==0)//XOZ_pos,XOZ_neg
    {
        numInterPlanes=2;
        interPlanesBool[2]=1; interPlanesBool[3]=1;
        ts[2]=(XOZ_y_pos - yca) / Dy;
        ts[3]=(XOZ_y_neg - yca) / Dy;
    }
    else if(Dy==0 && Dz==0)//YOZ_pos,YOZ_neg
    {
        numInterPlanes=2;
        interPlanesBool[4]=1; interPlanesBool[5]=1;
        ts[4]=(YOZ_x_pos - xca) / Dx;
        ts[5]=(YOZ_x_neg - xca) / Dx;
    }
    else//not parallel, then decide planes
    {
        //XOY_pos
        if(Dz!=0)
        {
            t = (XOY_z_pos - zca)/Dz;
            x = xca+t*Dx, y = yca+t*Dy;
            if (x>YOZ_x_neg && x<YOZ_x_pos && y>XOZ_y_neg && y<XOZ_y_pos){
                ts[0] = t;
                interPlanesBool[0] = 1;}
        }
        //XOY_neg
        if(Dz!=0)
        {
            t = (XOY_z_neg - zca)/Dz;
            x = xca+t*Dx, y = yca+t*Dy;
            if (x>YOZ_x_neg && x<YOZ_x_pos && y>XOZ_y_neg && y<XOZ_y_pos){
                ts[1] = t;
                interPlanesBool[1] = 1;}
        }
        //XOZ_pos
        if(Dy!=0)
        {
            t = (XOZ_y_pos - yca)/Dy;
            x = xca+t*Dx, z = zca+t*Dz;
            if (x>YOZ_x_neg && x<YOZ_x_pos && z>XOY_z_neg && z<XOY_z_pos){
                ts[2] = t;
                interPlanesBool[2] = 1;}
        }
        //XOZ_neg
        if(Dy!=0)
        {
            t = (XOZ_y_neg - yca)/Dy;
            x = xca+t*Dx, z = zca+t*Dz;
            if (x>YOZ_x_neg && x<YOZ_x_pos && z>XOY_z_neg && z<XOY_z_pos){
                ts[3] = t;
                interPlanesBool[3] = 1;}
        }
        //YOZ_pos
        if(Dx!=0)
        {
            t = (YOZ_x_pos - xca)/Dx;
            y = yca+t*Dy, z = zca+t*Dz;
            if (y>XOZ_y_neg && y<XOZ_y_pos && z>XOY_z_neg && z<XOY_z_pos){
                ts[4] = t;
                interPlanesBool[4] = 1;}
        }
        //YOZ_neg
        if(Dx!=0)
        {
            t = (YOZ_x_neg - xca)/Dx;
            y = yca+t*Dy, z = zca+t*Dz;
            if (y>XOZ_y_neg && y<XOZ_y_pos && z>XOY_z_neg && z<XOY_z_pos){
                ts[5] = t;
                interPlanesBool[5] = 1;}
        }
    }
    for (int ii=0;ii<6;ii++)
    {
        numInterPlanes += interPlanesBool[ii];
    }
    //interCubePlanes (end)
//    if(numInterPlanes==2)
//        printf("numInterPlanes=%d, ", numInterPlanes);
    if(numInterPlanes != 2){
        gLevel = 0;
    }else{
        //find out the two plane indexs
        //printf("pixel (%d,%d) is calculated.",i,j);
//        set_cvs_data_GPU(cvs_data_GPU, 255, cvs_width, i, j);
//        return;
//        printf("%d",psp);
        int planeIdx[2];
        int idxx = 0;
        for(int ii=0;ii<6;ii++)
        {
            if(interPlanesBool[ii] == 1){
                planeIdx[idxx] = ii;
                idxx++;
            }
        }
        //find out corresponding t1, t2
        double t1 = ts[planeIdx[0]];
        double t2 = ts[planeIdx[1]];
        //printf("rendering with t1=%.4f, t2=%.4f\n", t1, t2);
        //make sure that t1 < t2
        if(t1>t2){double temp=t1;t1=t2;t2=temp;}
        //show frame
        if(showFrame)
        {
            double x1 = xca+t1*Dx; double y1 = yca+t1*Dy; double z1 = zca+t1*Dz;
            double x2 = xca+t2*Dx; double y2 = yca+t2*Dy; double z2 = zca+t2*Dz;
            double xinc = xca+(tmin+tinc)*Dx, yinc = yca+(tmin+tinc)*Dy, zinc = zca+(tmin+tinc)*Dz;
            //printf("showFrame: x1=%.2f, x2=%.2f, xinc=%.2f, frameWidth=%.2f\n",x1,x2,xinc,frameWidth);
            if( ( sqrt(pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2)) <frameWidth) ||
                ( sqrt(pow((x1-xinc),2) + pow((y1-yinc),2) + pow((z1-zinc),2)) <frameWidth/6) ||
                ( sqrt(pow((xinc-x2),2) + pow((yinc-y2),2) + pow((zinc-z2),2)) <frameWidth/6)
                )
            {
                gLevel = frameColor;
                //printf("showFrame at (%d,%d): gLevel=%d\n",i,j,gLevel);
                set_cvs_data_GPU(cvs_data_GPU, 255, cvs_width, i, j);
                return;
            }
        }
        //perspection method
        switch((int)psp)
        {
        case(1):
        {
            double x1,y1,z1;
            //find out if is sliced
            if(t1>=tmin+tinc)//not sliced
            {
                //calculate coordinate of the nearest intersection point
                x1 = xca+t1*Dx; y1 = yca+t1*Dy; z1 = zca+t1*Dz;
            }else{
                //calculate coordinate of the nearest intersection point
                x1 = xca+(tmin+tinc)*Dx; y1 = yca+(tmin+tinc)*Dy; z1 = zca+(tmin+tinc)*Dz;
            }
            gLevel = getClosestValue(cb_array3D_head, x1, y1, z1,
                                     cb_xScale, cb_yScale, cb_zScale,
                                     cb_maxlength,
                                     cb_xshift, cb_yshift, cb_zshift,
                                     cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                                     cb_drawSliceBool, cb_mask, cb_status,
                                     cb_idx, cb_idy, cb_idz);
            break;
        }
        case(2):
        {
            gLevel = 0;
            for(t=max(tmin+tinc,t1);t<=t2;t=t+(tmax-tmin)*sample_step)
            {
                x = xca+t*Dx; y = yca+t*Dy; z = zca+t*Dz;
                uchar a = getClosestValue(cb_array3D_head, x, y, z,
                                     cb_xScale, cb_yScale, cb_zScale,
                                     cb_maxlength,
                                     cb_xshift, cb_yshift, cb_zshift,
                                     cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                                     cb_drawSliceBool, cb_mask, cb_status,
                                     cb_idx, cb_idy, cb_idz);
                if(a>gLevel)
                {
                    gLevel = a;
                }
            }
            break;
        }
        case(3):
        {
            double sum = 255;
            int count = 0;
            for(t=max(tmin+tinc,t1);t<=t2;t=t+(tmax-tmin)*sample_step)
            {
                x = xca+t*Dx; y = yca+t*Dy; z = zca+t*Dz;
                sum += getClosestValue(cb_array3D_head, x, y, z,
                                     cb_xScale, cb_yScale, cb_zScale,
                                     cb_maxlength,
                                     cb_xshift, cb_yshift, cb_zshift,
                                     cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                                     cb_drawSliceBool, cb_mask, cb_status,
                                     cb_idx, cb_idy, cb_idz);
                count++;
            }
            sum = sum/count;
            gLevel = uchar(sum);
            break;
        }
        case(0):
        {
            //this EMAB is simplified that "absorption_rate" are the same for all volumns

            //printf("(%d,%d) is running into EMAB rendering with t1=%.2f t2=%.2f...\n",i,j,t1,t2);
//            set_cvs_data_GPU(cvs_data_GPU, 255, cvs_width, i, j);
//            return;
            double C = 0;
            double transmissionRate_accuminate = 1;
            for(double tt=max(tmin+tinc,t1);tt<=t2;tt=tt+(tmax-tmin)*sample_step)
            {
                x = xca+tt*Dx; y = yca+tt*Dy; z = zca+tt*Dz;
                //printf("EMAB at (%.2f,%.2f,%.2f)\t",x,y,z);
                //printf("EMAB\t");
                uchar g = getClosestValue(cb_array3D_head, x, y, z,
                                     cb_xScale, cb_yScale, cb_zScale,
                                     cb_maxlength,
                                     cb_xshift, cb_yshift, cb_zshift,
                                     cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                                     cb_drawSliceBool, cb_mask, cb_status,
                                     cb_idx, cb_idy, cb_idz);
                //printf("tt=%.4f, x=%.4f, y=%.4f, z=%.4f\n, g=%d",tt,x,y,z,g);
                absorption_rate = double(g)/255. * int(g>=transparentThreshold_low) * int(g<=transparentThreshold_high) * transparent_reduce;
                C += double(g) * absorption_rate * transmissionRate_accuminate;
                transmissionRate_accuminate = transmissionRate_accuminate*(1-absorption_rate);
            }
            if(C>255)   C=255;
            //printf("C=%d",C);
            gLevel = uchar(C);
            break;
        }
        default:
        {
            //calculate coordinates of the two intersection points
            double x1 = xca+t1*Dx; double y1 = yca+t1*Dy; double z1 = zca+t1*Dz;
            double x2 = xca+t2*Dx; double y2 = yca+t2*Dy; double z2 = zca+t2*Dz;
            gLevel = getClosestValue(cb_array3D_head, x1, y1, z1,
                                     cb_xScale, cb_yScale, cb_zScale,
                                     cb_maxlength,
                                     cb_xshift, cb_yshift, cb_zshift,
                                     cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                                     cb_drawSliceBool, cb_mask, cb_status,
                                     cb_idx, cb_idy, cb_idz);
            break;
        }
        }
    }
    set_cvs_data_GPU(cvs_data_GPU, gLevel, cvs_width, i, j);
    return;
    //printf("\n");
}

void traceRayGPU(uchar *cvs_data,
                const int cvs_height, const int cvs_width,
                const double vpt_witdh, const double vpt_height,
                double xc, double yc, double zc,double beta,
                double xca, double yca, double zca,
                //intercubes
                double cb_xScale, double cb_yScale, double cb_zScale,
                double cb_xlength, double cb_ylength, double cb_zlength, double cb_maxlength,
                double cb_xshift, double cb_yshift, double cb_zshift,
                double cb_centerx, double cb_centery, double cb_centerz,
                double XOY_z_pos, double XOY_z_neg,
                double XOZ_y_pos, double XOZ_y_neg,
                double YOZ_x_pos, double YOZ_x_neg,
                bool cb_drawSliceBool,
                int cb_idx, int cb_idy, int cb_idz,
                uchar cb_mask, bool *cb_status,
                uchar *cb_array3D_head,int cb_array3D_sizex, int cb_array3D_sizey, int cb_array3D_sizez,
                //traceray
                double tmin,double tmax,double tinc,
                int psp,
                bool showFrame,double frameWidth,uchar frameColor,
                double sample_step,double absorption_rate,double transparent_reduce,
                double transparentThreshold_low, double transparentThreshold_high)
{
//    for(int ii=0;ii<cb_array3D_sizex;ii++){
//        for(int jj=0;jj<cb_array3D_sizey;jj++){
//            for(int kk=0;kk<cb_array3D_sizez;kk++){
//                uchar rr=cb_array3D_head[ii*(cb_array3D_sizey*cb_array3D_sizez)+(jj*cb_array3D_sizez+kk)];
//                if(rr!=0)
//                    printf("array[%d,%d,%d]=%d\t",ii,jj,kk,rr);
//            }
//        }
//    }
    //cvs_data_GPU
    uchar *cvs_data_GPU;
    cudaMalloc((uchar**)&cvs_data_GPU, sizeof(uchar)*cvs_height*cvs_width);
    //cb_array3D_head_GPU
    uchar *cb_array3D_head_GPU;
    int cb_array3D_head_GPU_size = cb_array3D_sizex*cb_array3D_sizey*cb_array3D_sizez;
    cudaMalloc((uchar**)&cb_array3D_head_GPU, sizeof(uchar)*cb_array3D_head_GPU_size);
    cudaMemcpy(cb_array3D_head_GPU, cb_array3D_head, sizeof(uchar)*cb_array3D_head_GPU_size, cudaMemcpyHostToDevice);
    //cb_status_GPU
    bool *cb_status_GPU;
    cudaMalloc((bool**)&cb_status_GPU, sizeof(bool)*3);
    cudaMemcpy(cb_status_GPU, cb_status, sizeof(bool)*3, cudaMemcpyHostToDevice);

    int gridNum = 1024;
    int blockNum = cvs_height*cvs_width/gridNum + 1;
    traceSinglePixel<<<gridNum,blockNum>>>(cvs_data_GPU,
                         cvs_height,cvs_width,
                         vpt_witdh,vpt_height,
                         xc,yc,zc,beta,
                         xca,yca,zca,
                         //intercubes
                         cb_xScale, cb_yScale, cb_zScale,
                         cb_xlength, cb_ylength, cb_zlength, cb_maxlength,
                         cb_xshift, cb_yshift, cb_zshift,
                         cb_centerx, cb_centery, cb_centerz,
                         XOY_z_pos, XOY_z_neg,
                         XOZ_y_pos, XOZ_y_neg,
                         YOZ_x_pos, YOZ_x_neg,
                         cb_drawSliceBool,
                         cb_idx, cb_idy, cb_idz,
                         cb_mask, cb_status_GPU,
                         cb_array3D_head_GPU, cb_array3D_sizex, cb_array3D_sizey, cb_array3D_sizez,
                         //traceray
                         tmin, tmax, tinc,
                         psp,
                         showFrame, frameWidth, frameColor,
                         sample_step, absorption_rate, transparent_reduce,
                         transparentThreshold_low, transparentThreshold_high);
    cudaDeviceSynchronize();

    cudaMemcpy(cvs_data, cvs_data_GPU, sizeof(uchar)*cvs_height*cvs_width, cudaMemcpyDeviceToHost);
    //printf("memcpy GPU2CPU successful!\n");

    //cvs_data_GPU
    cudaFree(cvs_data_GPU);
    //cb_array3D_head_GPU
    cudaFree(cb_array3D_head_GPU);
    //cb_status_GPU
    cudaFree(cb_status_GPU);
    //printf("cudaFree successful!\n");

    //printf("rayTracecGPU successful!\n");
    return;
}
