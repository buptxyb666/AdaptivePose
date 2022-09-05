pip3 install -i https://pypi.douban.com/simple/ numpy
pip3 install -i https://pypi.douban.com/simple/ torch==1.2.0
pip3 install -i https://pypi.douban.com/simple/ torchvision==0.4.0
pip3 install -i https://pypi.douban.com/simple/ opencv-python
pip3 install -i https://pypi.douban.com/simple/ Cython
pip3 install -i https://pypi.douban.com/simple/ numba
pip3 install -i https://pypi.douban.com/simple/ progress
pip3 install -i https://pypi.douban.com/simple/ matplotlib
pip3 install -i https://pypi.douban.com/simple/ easydict
pip3 install -i https://pypi.douban.com/simple/ scipy
pip3 install -i https://pypi.douban.com/simple/ pillow==6.2.1
pip3 install -i https://pypi.douban.com/simple/ scikit-image


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
