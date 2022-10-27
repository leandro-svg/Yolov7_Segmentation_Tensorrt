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
from carbox import BOX_3D

from utils.datasets import letterbox
from torchvision import transforms
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

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
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger,'')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def PreProcess(self, image_path):
        image = cv2.imread(image_path)
        real_image = image.copy()
        img = letterbox(image, self.imgsz, stride=64, auto=True)[0]
        img = transforms.ToTensor()(img)
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
            box3d = BOX_3D()
            for one_mask, conf, cls in zip(pred_masks_np, pred_conf, pred_cls):
                outfile = open('texte.txt', 'a')
                cnimg[:,:] = cnimg[:,:]*0
                if conf < 0.25:
                    continue
                label = '%s %.3f' % (names[int(cls)], conf)
                outfile.write(str(label)+'\n')
                outfile.close() 

                color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]                           
                pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
                cnimg[one_mask] = cnimg[one_mask]*0 + 255 
                nimg[one_mask] = nimg[one_mask]*0 + 255
                #cv2.imwrite("results/car_images/staging/one_mask_2/onemask"+str(ite)+".jpg", cnimg)
                if not(args.no_box) and ite < 40:
                        real_image = box3d.Box3D(pnimg, real_image, cnimg, args.save_image, args.cam, True, "VP_calibration")
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

        if args.use_trt :
            imh_path_alone = "/home/nvidia/SSD/Leandro_Intern/yolov7_trt/data/horses.jpg"
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
                img = letterbox(img_, self.imgsz, stride=64, auto=True)[0]
                img = transforms.ToTensor()(img)
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
                    print(" Saved in : " + "results/car_images/"+args.cam+"/segmentation/"+str(int(self.imgsz[0]))+"_trt_cv2img_VP_"+str(iteration)+".jpg")
                    # cv2.imwrite("results/car_images/"+args.cam+"/boxes/"+str(int(self.imgsz[0]))+"images_box_3D"+str(iteration)+".jpg", real_image)
                    # cv2.imwrite("results/car_images/"+args.cam+"/segmentation/"+str(int(self.imgsz[0]))+"_trt_cv2img_VP_"+str(iteration)+".jpg", pnimg)
                    # cv2.imwrite("results/car_images/"+args.cam+"/mask/"+str(int(self.imgsz[0]))+"_trt_cv2img_mask_VP_"+str(iteration)+".jpg", nimg)
                iteration += 1

def get_parser():        
    parser = argparse.ArgumentParser(
            description="Detectron2 demo for builtin models")
    parser.add_argument(
    "--input",
    default="/home/nvidia/SSD/Leandro_Intern/yolov7_trt/data/horses.jpg",
    nargs="+",
    help="A file or directory of your input data "
    "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
    "--model",
    default='./engine/640_yolov7-mask_fp32.engine',
    help="A file or directory of your model ",
    )
    parser.add_argument(
    "--imgsz",
    default=640,
    type=int,
    help="A file or directory of your model ",
    )
    parser.add_argument(
    "--use_trt",
    action="store_true",
    )
    parser.add_argument(
    "--save_image",
    action="store_true",
    )
    parser.add_argument(
    "--save_path",
    default="./results/car_images/",
    help="A file or directory of your output images ",
    )
    parser.add_argument(
        "--cam",
        default="None",
        type=str,
        help="staging, haunter or hypno",
        )
        
    parser.add_argument(
        "--no_box",
        action="store_true",
        )
    return parser

args = get_parser().parse_args()
arg_input = args.input
pred = BaseEngine(engine_path=args.model, imgsz=(args.imgsz,args.imgsz))
origin_img = pred.inference(arg_input)

