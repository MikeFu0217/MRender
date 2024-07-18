QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
CONFIG += static

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    widget.cpp
HEADERS += \
    array2D.h \
    array3D.h \
    canvas.h \
    imagecube.h \
    rayTrace_GPU.h \
    scene.h \
    test_CUDA.h \
    vec3D.h \
    viewport.h \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    rayTrace_GPU.cu \
    test_CUDA.cu

RC_FILE += logo.rc


# CUDA settings begin -----------------------------------------------------------------------------------------------
INCLUDEPATH +=D:\Applications\Qt\CUDA\v11.3\include  \
 D:\Applications\Qt\CUDA\v11.3\common\inc #Qt Creator配置路径不能包含空格，因此我把安装好的cuda新建了个文件夹
LIBS +=-LD:/Applications/Qt/CUDA/v11.3/lib/x64 \
-lcudart
OTHER_FILES +=./rayTrace_GPU.cu #要运行的程序
OTHER_FILES +=./test_CUDA.cu #要运行的程序
# Cuda sources
CUDA_SOURCES+=./rayTrace_GPU.cu
CUDA_SOURCES+=./test_CUDA.cu
CUDA_SDK ="D:/Applications/Qt/CUDA/v11.3"
CUDA_DIR ="D:/Applications/Qt/CUDA/v11.3"
QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
SYSTEM_TYPE = 64
#不同的显卡注意填适当的选项
CUDA_ARCH = sm_75
NVCCFLAGS     = --use_fast_math
CUDA_INC = $$join("D:/Applications/Qt/CUDA/v11.3/include",'" -I"','-I"','"')

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = "/MDd" #表示使用DLL的调试版
MSVCRT_LINK_FLAG_RELEASE = "/MD" #使用DLL版的C和C++运行库 具体可以看vs的属性设置

CUDA_OBJECTS_DIR = ./
# 配置编译器
CONFIG(debug, debug|release) {
# Debug mode
cuda_d.input = CUDA_SOURCES
cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}rayTrace_GPU.obj #windows的中间文件是obj格式，这个是需要修改d
cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE \
                 -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
cuda_d.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
# Release mode
 cuda.input = CUDA_SOURCES
 cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}rayTrace_GPU.obj
 cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE \
    -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
cuda.dependency_type = TYPE_C
 QMAKE_EXTRA_COMPILERS += cuda
}
# 配置编译器
CONFIG(debug, debug|release) {
# Debug mode
cuda_d.input = CUDA_SOURCES
cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}test_CUDA.obj #windows的中间文件是obj格式，这个是需要修改d
cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE \
                 -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
cuda_d.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
# Release mode
 cuda.input = CUDA_SOURCES
 cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}test_CUDA.obj
 cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE \
    -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
cuda.dependency_type = TYPE_C
 QMAKE_EXTRA_COMPILERS += cuda
}

RESOURCES += \
    resources.qrc
