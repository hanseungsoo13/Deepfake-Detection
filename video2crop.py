#%%
from __future__ import print_function
import os
import cv2
import torch
import argparse
import numpy as np
from glob import glob
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

import time
import easydict
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
#%%
args = easydict.EasyDict({
    'trained_model' : './weights/Resnet50_Final.pth',
    'network' : 'resnet50',
    'cpu' : False,
    'confidence_threshold' : 0.02,
    'top_k' : 5000,
    'nms_threshold' : 0.4,
    'keep_top_k' : 750,
    'vis_thres' : 0.6 })

device = torch.device("cpu" if args.cpu else "cuda")
#%%
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def network_settings():
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    net = net.to(device)
    return cfg, net

def detect(cfg, net, image):
    # testing begin
    # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(image)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass

    resize=1

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        cropped_img = image[b[1]: b[3], b[0]: b[2]]
        # save image
        return cropped_img
#%%
def video2crop(invideofilename, save_path):
    vidcap = cv2.VideoCapture(invideofilename)
    count = 0

    os.makedirs(save_path, exist_ok=True)

    while True:
        success, image = vidcap.read()
        if not success:
            break
            # print ('Read a new frame: ', success)
    
        fname = "\\{}.jpg".format("{0:05d}".format(count))
        cropped_img = detect(cfg, net, image)
        try:
            cv2.imwrite(save_path + fname, cropped_img)
            count += 1
        except:
            continue
    # print("{} images are extracted in {}.". format(count, save_path))
#%%
image_base_path = '..\\origin_dataset'
save_base_path = '..\\cropped_images'
data_list = [x for x in glob(image_base_path + '\\*') if not 'zip' in x]

cfg, net = network_settings()

for data in [data_list[1]]:
  data_name = Path(data).stem
  print('------------ {} ing ----------------'.format(data_name))
  folder_list = glob(data + '\\*')
  for folder in folder_list:
    folder_name = Path(folder).stem
    save_folder = os.path.join(save_base_path, data_name, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    print('>>>> {}'.format(folder_name))
    video_list = natsorted(glob(folder + '\\*'))
    for video_path in tqdm(video_list):
        video_id = Path(video_path).stem
        save_path = os.path.join(save_folder, video_id)
        video2crop(video_path, save_path)

# %%
image_base_path = '..\\origin_dataset'
save_base_path = '..\\cropped_images'
data_list = [x for x in glob(image_base_path + '\\*') if not 'zip' in x]

cfg, net = network_settings()

for data in [data_list[1]]:
  data_name = Path(data).stem
  print('------------ {} ing ----------------'.format(data_name))
  folder_list = glob(data + '\\*')
  for folder in folder_list:
    category_list = glob(folder + '\\*')

    for cat in category_list:
        cat_name = Path(cat).stem
        for root, dirs, files in os.walk(cat):
            if len(files) != 0:
                root_split = root.split('\\')
                root_split[1] = 'cropped_images'
                save_folder = '\\'.join(root_split)
                os.makedirs(save_folder, exist_ok=True)

                print('>>>> {}'.format(cat_name))
                video_list = natsorted(glob(root + '\\*'))
                
                for video_path in tqdm(video_list):
                    video_id = Path(video_path).stem
                    save_path = os.path.join(save_folder, video_id)
                    video2crop(video_path, save_path)
# %%
