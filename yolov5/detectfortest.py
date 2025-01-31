# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import math
import pandas as pd
import time

import torch
import csv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    
print(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode



# 두점 사이의 거리 구하는 함수
def distance(x, y):
    result = math.sqrt( math.pow(0.5 - x, 2) + math.pow(1.0 - y, 2))
    return result

# 점형블록 2개이상 나왔을때 좌/우/T/+ 교차 식별하는 함수
def discrimination(det,gn,frame):
    idx=0
    df_frame= pd.DataFrame({'frame':[],'block_type':[], 'center_x':[],'center_y':[],'width':[],'height':[]})
    
    #dataframe으로 변경
    for *xyxy, conf, cls in reversed(det):                  
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        df_frame.loc[idx]=[frame,cls.item(),xywh[0],xywh[1],xywh[2],xywh[3]]
        idx=idx+1
        
    #center_y 로 오름차순정렬 
    df_frame = df_frame.sort_values('center_y', ascending = True)
    
    #y값이 작은 점형 두개 뽑음
    df_dotTop = df_frame[df_frame['block_type'] == 2].head(2)
    
    #x값이 0.5와 가까운점을 기준 점형 블록으로 설정
    if abs(0.5-df_dotTop['center_x'].values[0])<=abs(0.5-df_dotTop['center_x'].values[1]): 
        std_x = df_dotTop['center_x'].values[0]
        std_y = df_dotTop['center_y'].values[0]
    else:
        std_x = df_dotTop['center_x'].values[1]
        std_y = df_dotTop['center_y'].values[1]
        
    #센터 점형좌표랑 기준점(0.5,1)사이의 거리
    std_dis = distance(std_x,std_y)
    
    left=0
    right=0
    go=0
    
    for row in df_frame[df_frame['block_type'] == 3].itertuples(): #선형블록 한 행씩
        #해당 선형블록과 중심거리
        line_dis = distance(row.center_x,row.center_y) 
        
        #선형블록 거리가 기준점형블록 거리보다 큰경우
        if line_dis>=std_dis: 
            if row.center_y <= (std_y - 0.04): 
                go = go + 1
            elif row.center_x >= std_x:            
                right=right+1
            elif row.center_x <= std_x:            
                left=left+1
        
    if right >= 1 and left >= 1 and go>=1: # + 교차
        return "plus"
    elif right >= 1 and left >= 1 : # T 교차
        return "t"
    elif right >= 1 and go>=1 : # 직진앤우회전
        return "rNg"
    elif left >= 1 and go>=1 : # 직진앤좌회전
        return "lNg"
    elif right >= 1 : # 우회전
        return "right"
    elif left >= 1 : #좌회전
        return "left"
        
    return "dot"    

    
@smart_inference_mode()
def run(
        weights=ROOT / '/home/jet/Desktop/yolov5/best.pt',  # model path or triton URL
        source= 0,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(416, 416),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #dataset을 LoadImages 클래스를 통해 정의
    vid_path, vid_writer = [None] * bs, [None] * bs
     
    ### 연속 갯수 딕셔너리 선언 ###
    typeCountDic = {'Dotted':0, 'Linear':0, 'Crosswalk':0, 'Bollard':0, 'Right':0, 'Left':0, 'T':0, 'Plus':0, 'lNg':0, 'rNg':0}
    ##Whether first go straight message
    firstLinearMsg = 0
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    #디바이스에는 디스플레이가 없기 때문에 시스템이 시작되었다는 음성안내 출력
    #이전 시간
    prevTime = time.time()
    num_iterations = 1
    total_time=0
    s_conf= 0                    
    nnn=0
    for path, im, im0s, vid_cap, s in dataset:
      	#프레임 수 계산
        curTime = time.time()   #현재 시간
        output_fps = 1 / (curTime - prevTime)
        test = curTime - prevTime
        total_time+=test
        num_iterations+=1
        prevTime = curTime
            
        output_fps_str = "FPS : %0.1f" %output_fps
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'Frame : {seen}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            dot = 0
            line = 0
            bollard=0
            crosswalk=0
            

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    if names[int(c)] == "Dotted":
                        dot = n.item()
                    elif names[int(c)] == "Linear":
                        line = n.item()
                    elif names[int(c)] == "Crosswalk":
                        crosswalk = n.item()
                    elif names[int(c)] == "Bollard":
                        bollard = n.item()
                      
                # 점형블록 2개 이상 검출 시
                if dot>=2:
                    result = discrimination(det,gn,frame)
                    if result == "plus":
                        print("+")
                        typeCountDic['Plus'] = typeCountDic['Plus'] + 1
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Left'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                    elif result == "t":
                        print("t")
                        cv2.putText(im0, "T Cross", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        typeCountDic['T'] = typeCountDic['T'] + 1
                        typeCountDic['Plus'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Left'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                    elif result == "right":
                        print("r")
                        typeCountDic['Right'] = typeCountDic['Right'] + 1
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Left'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                    elif result == "left":
                        print("l")
                        typeCountDic['Left'] = typeCountDic['Left'] + 1
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                    elif result == "rNg":
                        print("rg")
                        typeCountDic['Left'] = 0
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = typeCountDic['rNg'] + 1
                        typeCountDic['lNg'] = 0
                    elif result == "lNg":
                        print("lg")
                        typeCountDic['Left'] = 0
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Dotted'] = 0
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = typeCountDic['lNg'] + 1
                    elif dot>=3:
                        print("dd")
                        typeCountDic['Dotted'] = typeCountDic['Dotted'] + 1
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Left'] = 0                        
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                    else :
                        typeCountDic['Dotted'] = 0
                        typeCountDic['Plus'] = 0
                        typeCountDic['T'] = 0
                        typeCountDic['Right'] = 0
                        typeCountDic['Left'] = 0                        
                        typeCountDic['rNg'] = 0
                        typeCountDic['lNg'] = 0
                else:
                    typeCountDic['Dotted'] = 0
                    typeCountDic['Plus'] = 0
                    typeCountDic['T'] = 0
                    typeCountDic['Right'] = 0
                    typeCountDic['Left'] = 0
                    typeCountDic['rNg'] = 0
                    typeCountDic['lNg'] = 0
                    # firstLinearMsg == 0
                
                # 횡단보도
                if crosswalk>=1:
                    typeCountDic['Crosswalk'] = typeCountDic['Crosswalk'] + 1
                else:
                    typeCountDic['Crosswalk'] = 0

                # 볼라드
                if bollard>=1:
                    typeCountDic['Bollard'] = typeCountDic['Bollard'] + 1
                else:
                    typeCountDic['Bollard'] = 0
                    
                #선형블록 3개 이상 검출 시
                if line>=3:
                    typeCountDic['Linear'] = typeCountDic['Linear'] + 1
                else:
                    typeCountDic['Linear'] = 0
                    firstLinearMsg = 0

                if  ( typeCountDic['Crosswalk'] >= 5 or typeCountDic['Bollard'] >= 5 or typeCountDic['Dotted'] >= 5 
                        or typeCountDic['T'] >= 5 or typeCountDic['Plus'] >= 5 or typeCountDic['Right'] >= 5 
                        or typeCountDic['Left'] >= 5 or typeCountDic['lNg'] >= 5 or typeCountDic['rNg'] >= 5 ) :
                             
                    if typeCountDic['Crosswalk'] >= 5 and typeCountDic['Bollard'] >= 5:
                        print(".")
                    elif typeCountDic['Bollard'] >= 5:
                        print(".")
                    elif typeCountDic['Crosswalk'] >= 5:
                        print(".")

                    if typeCountDic['Plus'] >= 5:
                        print(".")
                    elif typeCountDic['T'] >= 5:
                        print(".")
                        #cv2.putText(im0, "T Cross", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    elif typeCountDic['rNg'] >= 5:
                        print(".")
                    elif typeCountDic['lNg'] >= 5:
                        print(".")
                    elif typeCountDic['Right']>=5:
                        print(".")
                    elif typeCountDic['Left']>=5:
                        print(".")
                    elif typeCountDic['Dotted']>=5:
                        print(".")
                        
                elif firstLinearMsg == 0 and typeCountDic['Linear']==15:
                    firstLinearMsg = 1

                elif firstLinearMsg == 1 and typeCountDic['Linear']>15:
                    if typeCountDic['Linear']%50 == 0:
                        print(".")
                	
            
                #cv2.putText(im0, output_fps_str , (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                #cv2.putText(im0, str(frame) , (100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    s_conf= s_conf+conf
                    nnn=nnn+1
                    print(s_conf, nnn)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(save_path+'loc_test.csv','a') as f: #.csv파일로 저장
                            csvWriter = csv.writer(f)
                            csvWriter.writerow([frame, int(cls), xywh[0], xywh[1], xywh[2], xywh[3], conf])
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            else:
                typeCountDic['Dotted'] = 0
                typeCountDic['Plus'] = 0
                typeCountDic['T'] = 0
                typeCountDic['Right'] = 0
                typeCountDic['Left'] = 0
                typeCountDic['rNg'] = 0
                typeCountDic['lNg'] = 0
                typeCountDic['Crosswalk'] = 0
                typeCountDic['Linear'] = 0
                typeCountDic['Bollard'] = 0
            
                #cv2.putText(im0, output_fps_str, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}xxx{' ' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

	#Per Image for Function End
    average_time = total_time/num_iterations
    average_fps=1/average_time
    a_conf = s_conf/nnn
    print("total_time : ", total_time)
    print("num_iterations: ", num_iterations)
    print("Average FPS: ", average_fps)
    print("sumconf : ", s_conf)
    print("countconf : ", nnn)
    print("averageconf : ", a_conf)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= ROOT / '/home/jet/Desktop/yolov5/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default= 0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
