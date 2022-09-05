from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .exdet import ExdetDetector
# from .ddd import DddDetector
# from .ctdet import CtdetDetector
# from .multi_pose import MultiPoseDetector
from .multi_pose_wodet import MultiPoseDetector_wodet
from .multi_pose_crowdpose import MultiPoseDetector_crowdpose

detector_factory = {
  # 'exdet': ExdetDetector, 
  # 'ddd': DddDetector,
  # 'ctdet': CtdetDetector,
  # 'multi_pose': MultiPoseDetector, 
  'multi_pose_wodet': MultiPoseDetector_wodet,
  'multi_pose_crowdpose': MultiPoseDetector_crowdpose
}
