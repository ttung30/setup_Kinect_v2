# Set up Camera Kinect v2
# cài đặt các gói phụ thuộc
sudo apt-get install build-essential cmake pkg-config  -y  # Build tools
sudo apt-get install libusb-1.0-0-dev          # libusb
sudo apt-get install libturbojpeg0-dev -y      # TurboJPEG
sudo apt-get install libglfw3-dev -y           # OpenGL
sudo apt-get install libva-dev libjpeg-dev -y  # VAAPI for Intel only
sudo apt-get install libopenni2-dev -y         # OpenNI2

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

