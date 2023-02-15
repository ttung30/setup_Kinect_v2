# Set up Camera Kinect v2
$ git clone https://github.com/OpenKinect/libfreenect2
$ cd libfreenect2
# Tạo thư mục build
$ mkdir build && cd build
# Biên dịch
$ cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
$ make
$ make install 
# Fix lỗi biên dịch 
https://github.com/OpenKinect/libfreenect2/issues/777#issuecomment-480023146
# Install pylibfreenect2 && hướng dẫn
https://r9y9.github.io/pylibfreenect2/latest/installation.html

