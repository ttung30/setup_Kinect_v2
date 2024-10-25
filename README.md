# Set up Camera Kinect v2
## Dependence pakages
```bash
sudo apt-get install build-essential cmake pkg-config  -y  
sudo apt-get install libusb-1.0-0-dev
sudo apt-get install libturbojpeg0-dev -y     
sudo apt-get install libglfw3-dev -y  
sudo apt-get install libva-dev libjpeg-dev -y 
sudo apt-get install libopenni2-dev -y 
```
## install Kinect library
```bash
$ git clone https://github.com/OpenKinect/libfreenect2
$ cd libfreenect2
$ mkdir build && cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
$ make
$ make install 
```
## Fix install Kinect library
```bash
https://github.com/OpenKinect/libfreenect2/issues/777#issuecomment-480023146
```
## Install python library for developing
```bash
https://r9y9.github.io/pylibfreenect2/latest/installation.html
```

