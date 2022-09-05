from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .sample.multi_pose_wodet import MultiPoseDataset_wodet
from .sample.multi_pose_crowdpose import MultiPoseCrowdpose


from .dataset.coco_hp_wodet import COCOHP_wodet
from .dataset.crowdpose import CrowdPose


dataset_factory = {
  'coco_hp_wodet': COCOHP_wodet,
  'crowdpose': CrowdPose
}

_sample_factory = {
  'multi_pose_wodet': MultiPoseDataset_wodet,
  'multi_pose_crowdpose': MultiPoseCrowdpose
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
