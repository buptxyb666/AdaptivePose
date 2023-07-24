pip3 install numpy
pip3 install torch
pip3 install torchvision
pip3 install opencv-python
pip3 install Cython
pip3 install numba
pip3 install progress
pip3 install matplotlib
pip3 install easydict
pip3 install scipy
pip3 install pillow==6.2.1
pip3 install scikit-image


THIS_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
cd $THIS_DIR

CURRENT_DIR=$(pwd)

# cd $CURRENT_DIR
# source prepare_env.sh
cd $CURRENT_DIR/cocoapi/PythonAPI
make
python3 setup.py install --user


cd $CURRENT_DIR/CrowdPose/crowdpose-api/PythonAPI
make install
python3 setup.py install --user


cd $CURRENT_DIR/src/lib/models/networks/DCNv2
python3 setup.py build develop
cd $CURRENT_DIR/src/lib/external
make
cd $CURRENT_DIR/src/lib/models/resample2d_package
python3 setup.py install --user
