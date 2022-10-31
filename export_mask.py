import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import argparse
import time
import numpy as np
import argparse
import onnxruntime as ort
import os
import torch
import torch.backends.cudnn as cudnn
import cv2
import onnx
import yaml
from torchvision import transforms
import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
from utils.general import set_logging
from models.experimental import attempt_load




def PostProcess(img, hyp, model, inf_out, attn, bases, sem_output):
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = img.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
    
    output, output_mask = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()


    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        print(color)       
                            
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return pnimg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument("--input", nargs="+", help="A file or directory of your input data ")
    parser.add_argument('--imgsz', type=int, default=320, help='image size')  # height, width
    parser.add_argument('--no_infer', action='store_true', help='CoreML FP16 half-precision export')

    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    set_logging()
    t = time.time()

    
    device = torch.device( "cpu")
   
    
    with open('data/hyp.scratch.mask.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device( "cpu")
    weights = opt.weights
    model = attempt_load(weights, map_location=device)
    _ =    model.eval()
   
    import time
    time1 = time.time()
    loop = 1
    for i in range(loop):
        image = cv2.imread('data/horses.jpg')  # 504x378 image
        image = letterbox(image, (opt.imgsz,opt.imgsz), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(device)
        img = image
    y = model(image)
    
    try:
        import onnx
    
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f ="./onnx/"+str(int(opt.imgsz))+"_yolov7_mask.onnx" # filename
        model.eval()
        output_names = ['output']
        dynamic_axes = None
        if opt.grid : 
            model.model[-1].concat = True
        torch.onnx.export(model, image, f, verbose=True, opset_version=13, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        onnx.save(onnx_model,f)
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


if not(opt.no_infer):
    f ="./onnx/"+str(int(opt.imgsz))+"_yolov7-_.onnx" # filename
    image_path = opt.input

    iteration = 0
    start_time_all = time.time()
    w = f
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)

    model_onnx = onnx.load(w)
    input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model_onnx.graph.input]
    output_shapes = [[d.dim_value for d in _output.type.tensor_type.shape.dim] for _output in model_onnx.graph.output]



    outname = [i.name for i in session.get_outputs()]

    inname = [i.name for i in session.get_inputs()]
    time_use_trt_only = 0
    time_use_trt_ = 0
    for img_path in tqdm.tqdm(image_path):
        start_time = time.time()
        image = cv2.imread(img_path)
        image = letterbox(image, (opt.imgsz, opt.imgsz), stride=64, auto=True)[0]
        image_letter = image.copy()
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))  ##tensor or numpy??
        img = np.array(image)
        
        img = np.ascontiguousarray(img, dtype=np.float32)
        inp = {inname[0]:img}
        output = session.run(outname, inp)[0]
        output1 = session.run(outname, inp)[1]
        output2 = session.run(outname, inp)[2]
        output3 = session.run(outname, inp)[3]
        output4 = session.run(outname, inp)[4]
        output5 = session.run(outname, inp)[5]
        output6 = session.run(outname, inp)[6]
        inf_out, train_out = torch.tensor(output), [torch.tensor(output2),torch.tensor(output3),torch.tensor(output4)]
        attn, mask_iou, bases, sem_output = torch.tensor(output1), None, torch.tensor(output5), torch.tensor(output6)
        img = torch.tensor(img)
        pnimg = PostProcess(img, hyp, model, inf_out, attn, bases, sem_output)

        
        save_path = "./result_onnx"
        cv2.imwrite(save_path+str(int(opt.imgsz))+".jpg", pnimg)
        iteration+=1