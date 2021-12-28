import sys
sys.path.insert(0, './object_tracking/deep_sort')

from object_tracking.deep_sort.utils.parser import get_config
from object_tracking.deep_sort.deep_sort import DeepSort

from object_detection.yolov5.utils.downloads import attempt_download
class Tracker:
    def __init__(self, path_config):
        self.cfg = get_config()
        self.cfg.merge_from_file(path_config)
        
        self.deepsort = DeepSort(
            self.cfg.DEEPSORT.MODEL_TYPE,
            max_dist=self.cfg.DEEPSORT.MAX_DIST, 
            min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.DEEPSORT.MAX_AGE, 
            n_init=self.cfg.DEEPSORT.N_INIT, 
            nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True)
        self.max_age = self.cfg.DEEPSORT.MAX_AGE
        
        # print('DEEPSORT', self.cfg)
