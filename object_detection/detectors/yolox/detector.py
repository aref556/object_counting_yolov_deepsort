import sys
import os
from loguru import logger

sys.insert(0, 'object_detection')

from object_detection.YOLOX.yolox.exp.build import get_exp
from object_detection.YOLOX.tools.demo import Predictor

class Detector(object):
    def __init__(
        self, 
        demo="image", # demo type eg. image. video and webcam
        experiment_name=None, # type str
        name=None, # model name
        path="./object_detection/YOLOX/assets/dog.jpg",
        camid=0, # webcam demo camera id
        save_result=None, # type boolean, whether to save the inference result of image/video
        exp_file=None, # tpye str, pls input your experiment description file
        ckpt=None, # type str, ckpt for eval
        device="cpu", # type str, device to run our model, can either be cpu or gpu
        conf=0.3, # type float, test conf
        nms=0.3, # type float, test nms threshold
        tsize=None, #  type int, test img size
        fp16=False, # type boolean, Adopting mix precision evaluating.
        legacy=False, # type boolean, To be compatible with older versions
        fuse=False, # type boolean, Fuse conv and bn for testing.
        trt=False, # type boolean, Using TensorRT model for testing.
        
    ):
        self.demo = demo
        self.experiment_name = experiment_name
        self.name = name
        self.path = path
        self.camid = camid
        self.save_result = save_result
        self.exp_file = exp_file
        self.ckpt = ckpt
        self.device = device
        self.conf = conf
        self.nms = nms
        self.tsize = tsize
        self.fp16 = fp16
        self.legacy = legacy
        self.fuse = fuse
        self.trt = trt
        
        self.exp = get_exp(self.exp_file, self.name)
        

    def detect(self):
        
        
        if not self.experiment_name:
            self.experiment_name = self.exp.exp_name

        file_name = os.path.join(self.exp.output_dir, self.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        vis_folder = None
        if self.save_result:
            vis_folder = os.path.join(file_name, "vis_res")
            os.makedirs(vis_folder, exist_ok=True)

        if self.trt:
            self.device = "gpu"

        logger.info("self: {}".format(self))

        if self.conf is not None:
            exp.test_conf = self.conf
        if self.nms is not None:
            exp.nmsthre = self.nms
        if self.tsize is not None:
            exp.test_size = (self.tsize, self.tsize)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if self.device == "gpu":
            model.cuda()
            if self.fp16:
                model.half()  # to FP16
        model.eval()

        if not self.trt:
            if self.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = self.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.trt:
            assert not self.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        predictor = Predictor(
            model, exp, COCO_CLASSES, trt_file, decoder,
            self.device, self.fp16, self.legacy,
        )
        current_time = time.localtime()
        if self.demo == "image":
            image_demo(predictor, vis_folder, self.path, current_time, self.save_result)
        elif self.demo == "video" or self.demo == "webcam":
            imageflow_demo(predictor, vis_folder, current_time, self)