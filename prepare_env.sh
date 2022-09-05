THIS_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
cd $THIS_DIR
tar zxvf torch12.tar.gz

cd $THIS_DIR
CURRENT_DIR=$(pwd)

# cd $CURRENT_DIR
# source prepare_env.sh
cd $CURRENT_DIR/cocoapi/PythonAPI
make
$CURRENT_DIR/torch12/bin/python setup.py install --user


cd $CURRENT_DIR/CrowdPose/crowdpose-api/PythonAPI
make install
$CURRENT_DIR/torch12/bin/python setup.py install --user


cd $CURRENT_DIR/src/lib/models/networks/DCNv2
$CURRENT_DIR/torch12/bin/python setup.py build develop
cd $CURRENT_DIR/src/lib/external
make
cd $CURRENT_DIR/src/lib/models/resample2d_package
$CURRENT_DIR/torch12/bin/python setup.py install --user
