# Set up Camera Kinect v2
$ git clone https://github.com/OpenKinect/libfreenect2
$ cd libfreenect2
# Tạo thư mục build
$ mkdir build && cd build
# Biên dịch
$ cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
$ make
$ make install 

