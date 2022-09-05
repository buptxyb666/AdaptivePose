from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .ctdet import CtdetTrainer
# from .ddd import DddTrainer
# from .exdet import ExdetTrainer
# from .multi_pose import MultiPoseTrainer
from .multi_pose_wodet import MultiPoseTrainer_wodet
from .multi_pose_crowdpose import MultiPoseTrainer_crowdpose 

train_factory = {
  # 'exdet': ExdetTrainer, 
  # 'ddd': DddTrainer,
  # 'ctdet': CtdetTrainer,
  # 'multi_pose': MultiPoseTrainer, 
  'multi_pose_wodet': MultiPoseTrainer_wodet,
  'multi_pose_crowdpose': MultiPoseTrainer_crowdpose
}
