import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import argparse
import os
import torch
import yaml
import tqdm
import glob
import onnx
import time
from pathlib import Path
import logging

from utils.datasets import letterbox
from torchvision import transforms
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode

def set_logging(name=None, verbose = VERBOSE):
    # Sets level and returns logger
    
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class BaseEngine(object):
    def __init__(self, engine_path, imgsz=(320,320), use_torch=False, use_onnx=False):
        set_logging("yolov5")  # run before defining LOGGER        
        self.LOGGER = logging.getLogger("yolov5")
        self.imgsz = imgsz
        self.mean = None
        self.std = None
        self.use_torch = use_torch
        self.use_onnx = use_onnx
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        f = args.onnx_model
        model_onnx = onnx.load(f)
        self.input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model_onnx.graph.input]

        self.LOGGER.info(f'Loading {engine_path} for TensorRT inference...')
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger,'')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.LOGGER.info('Deserializing engine...')
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            self.size_trt = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(self.size_trt, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        self.LOGGER.info('Engine Loaded.')
    def PreProcess(self, image_path):
        image = cv2.imread(image_path)
        real_image = image.copy()
        image = letterbox(image, self.imgsz, stride=64, auto=True)[0]
        if (np.shape(image) != self.input_shapes[0][2:4]): #Not the same shape as the input of the onnx model, needs to implement dynamical shape
            image = (cv2.resize(image, self.input_shapes[0][2:4]))
        img = transforms.ToTensor()(image)
        img = torch.unsqueeze(img, 0)
        return img, real_image

    def PostProcess(self,img, hyp, inf_out, attn, bases, sem_output, real_image):
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = img.shape
        pooler_scale = 0.25
        names = self.class_names
        pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        
        output, output_mask = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

        pred, pred_masks = output[0], output_mask[0]
        if pred is not None : 
           
            bboxes = Boxes(pred[:, :4])
            original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
            pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
            pred_masks_np = pred_masks.detach().cpu().numpy()
            pred_conf = pred[:, 4].detach().cpu().numpy()
            pred_cls = pred[:, 5].detach().cpu().numpy()
            nimg = img[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            pnimg = nimg.copy()
            nimg[:,:] = nimg[:,:]*0
            cnimg = nimg.copy()
            ite = 0
            for one_mask, conf, cls in zip(pred_masks_np, pred_conf, pred_cls):
                cnimg[:,:] = cnimg[:,:]*0
                if conf < 0.25:
                    continue
                label = '%s %.3f' % (names[int(cls)], conf)
                color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]                           
                pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
                cnimg[one_mask] = cnimg[one_mask]*0 + 255 
                nimg[one_mask] = nimg[one_mask]*0 + 255
                ite +=1
        else : 
            print("No predictions")
            nimg = img[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            pnimg = nimg.copy()
            nimg[:,:] = nimg[:,:]*0
            pnimg = nimg
            nimg = nimg
            real_image =  real_image
        return pnimg, nimg, real_image

    def infer(self, img):
        img = np.ascontiguousarray(img, dtype=np.float32)
        self.inputs[0].host = img
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        data = [out.host for out in self.outputs]
        return data

    def inference(self, image_path, conf=0.25):
        with open('data/hyp.scratch.mask.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)

        imh_path_alone = "./data/horses.jpg"
        img, real_image = self.PreProcess(imh_path_alone)
        for _ in  range(5):
            output = self.infer(img) #dry run
        
        iteration = 0
        if len(image_path) == 1:
            image_path = glob.glob(os.path.expanduser(image_path[0]))
            assert image_path, "The input path(s) was not found"
        for img_path in tqdm.tqdm(image_path):
            img_ = cv2.imread(img_path)
            real_image = img_.copy()
            image = letterbox(img_, self.imgsz, stride=64, auto=True)[0]
            if (np.shape(image)[0:2] != self.input_shapes[0][2:4]): #Not the same shape as the input of the onnx model, needs to implement dynamical shape
                print("/!\ Shape of the input " + str(np.shape(image)[0:2]) + " different from the input size of the ONNX model "+ str(self.input_shapes[0][2:4])+", have to resize the image.")
                image = (cv2.resize(image, (self.input_shapes[0][3], self.input_shapes[0][2])))
            img = transforms.ToTensor()(image)
            img = torch.unsqueeze(img, 0)

            output = self.infer(img)


            for i in range(len(output)):
                output[i] = torch.tensor(output[i])
            inf_out = torch.reshape((output[5]), (1, len((output[5]))//85,85))
            attn = torch.reshape((output[6]), (1, (len((output[6]))//980),980))
            bases = torch.reshape( (output[0]), (1, 4, ((len(output[0])//(self.imgsz[0]//4))//4), (self.imgsz[0]//4)))
            sem_output = torch.reshape((output[1]), (1, 1, (len(output[1])//(self.imgsz[0]//4)), (self.imgsz[0]//4)))
            pnimg, nimg, real_image = self.PostProcess(img, hyp, inf_out, attn, bases, sem_output, real_image)
            
            if args.save_image:
                Path(args.save_path).mkdir(parents=True, exist_ok=True)
                print(" Saved in : " + str(args.save_path)+str(int(self.imgsz[0]))+"_trt_cv2img_VP_"+str(iteration)+".jpg")
                cv2.imwrite(str(args.save_path)+str(int(self.imgsz[0]))+"_trt_cv2img_VP_"+str(iteration)+".jpg", pnimg)
            iteration += 1
         if args.save_image : 
            self.LOGGER.info('Results are saved in :'+args.save_path)
        self.LOGGER.info('Done.')
def get_parser():        
    parser = argparse.ArgumentParser(
            description="Detectron2 demo for builtin models")
    parser.add_argument(
    "--input",
    default="./data/horses.jpg",
    nargs="+",
    help="A file or directory of your input data "
    "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
    "--model",
    default='./engine/yolov7-mask.engine',
    help="A file or directory of your model ",
    )
    parser.add_argument(
    "--onnx_model",
    default='onnx/640_yolov7_mask.onnx',
    help="A file or directory of your onnx model ",
    )
    parser.add_argument(
    "--imgsz",
    default=640,
    type=int,
    help="A file or directory of your model ",
    )
    parser.add_argument(
    "--save_image",
    action="store_true",
    )
    parser.add_argument(
    "--save_path",
    help="A file or directory of your output images ",
    )
    return parser

args = get_parser().parse_args()
arg_input = args.input
if (args.save_path is None and args.save_image):
    print("You need a result directory : mkdir results && --save_path results/")
    exit(0)
pred = BaseEngine(engine_path=args.model, imgsz=(args.imgsz,args.imgsz))
pred.LOGGER.info('TensorRT inference begins !')
origin_img = pred.inference(arg_input)


