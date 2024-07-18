#ifndef RAYTRACE_GPU_H
#define RAYTRACE_GPU_H

typedef unsigned char uchar;

void traceRayGPU(uchar *cvs_data_GPU,
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
                double transparentThreshold_low, double transparentThreshold_high);

#endif // RAYTRACE_GPU_H
