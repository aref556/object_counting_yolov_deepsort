import time
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

# limit the number of cpus used by high performance libraries
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil

import sys
sys.path.insert(0, './object_detection')
from object_detection.detectors.yolov5.detector import Detector

# sys.path.insert(0, './object_detection/yolov5')
from object_detection.yolov5.utils.datasets import LoadImages, LoadStreams
from object_detection.yolov5.utils.general import check_imshow
from object_detection.yolov5.utils.torch_utils import time_sync
from object_detection.yolov5.utils.plots import Annotator, colors 
from object_detection.yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
                           
sys.path.insert(0, './object_tracking')
from object_tracking.tracker import Tracker

sys.path.insert(0, './implement')
from implement.analyzer import Analyzer
# from implement.bdeepsort import Bdeepsort
from implement.zpolygon import Zpolygon


PATH_TO_CONFIG_DEEPSORT = 'object_tracking/deep_sort/configs/deep_sort.yaml'
PATH_TO_CONFIG_ANALYZER = 'implement/configs/analyzer.yaml'
PATH_WEIGHT_YOLOV5 = 'object_detection/yolov5/weights/yolov5l6.pt'
PATH_SOURCE ='videos/t3.mkv'

# PATH_SAVE_MP4="outputs/t1.mp4"
PATH_SAVE_FOLDER = 'inference/output/torchreld'
SAVE_NAME = 's2_yolov5l6_polypoly_repair_resnet50'

def main():
    print('cuda is available ?', torch.cuda.is_available())
    # paremeter to run file main
    show_vid = True
    webcam = False
    source = PATH_SOURCE
    # save_path = PATH_SAVE_MP4
    save_vid = True
    visualize = False
    out = PATH_SAVE_FOLDER
    # evaluate = False
    polygons = [
        # [(400,500),(600,200), (800, 200),(750,500),(600,580)], 
        [(400, 500),(600, 200), (800, 200), (750, 500)], 
        [(790, 500), (830, 200), (1100,200), (1150, 500)], 
        [(1190, 500), (1130, 200), (1300,200), (1550, 500)],
        [(780, 900),(780, 600), (1200, 600), (1200, 900)]
    ]
    # # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder
    
    # create Detector is Yolov5 classes=(2,7)
    detector = Detector(ckpt=PATH_WEIGHT_YOLOV5, conf_thres=0.3, classes=(2, 5, 6, 7, 8))
    
    # create Tracker is Deepsort
    tracker = Tracker(path_config=PATH_TO_CONFIG_DEEPSORT)
    
    # create Rdeepsort is Manage another but cam change name develop in this class
    analyzer = Analyzer(path_config=PATH_TO_CONFIG_ANALYZER, max_age=tracker.max_age, polygons=polygons)
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        source = '0'
        dataset = LoadStreams(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    
    dt, seen = [0.0, 0.0, 0.0], 0
    
    
    
    t0 = time.time()
    # Start Process
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        time_start = time.time()
        t1 = time_sync()
        img = torch.from_numpy(img).to(detector.device)
        img = img.half() if detector.half else img.float()
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        visualize = increment_path(save_path / Path(path).stem, mkdir=True) if visualize else False
        t3 = time_sync()
        dt[1] += t3 - t2

        # Draw Polygon Definition 
        if analyzer.polygons is not None: # check from polygons because is None if have polygon
            for i in analyzer.polygon:
                polygon = analyzer.polygon[i]
                polygon.draw_polygon(img=im0s)
        
        pred, dt[1], dt[2], t3 = detector.detect(img=img, dt=dt, t2=t2, visualize=visualize)

        
        for i, det in enumerate(pred):
            seen += 1
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            s += '%gx%g' % img.shape[2:] # ex. print string
            name_file = SAVE_NAME + ".mp4"
            save_path = str(Path(out) / name_file) 
            
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {detector.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                xywhs = xyxy2xywh(det[:, 0:4])
                # print("xywhs  ", xywhs)
                confs = det[:, 4]
                confs_mod = confs.cpu()
                # print("confs  ", confs_mod)
                confs_mod = torch.tensor([max(confs_mod)]) 
                            
                # print("confs mod  ", confs_mod)
                # ...
                clss = det[:, 5]
                
                clss_mod = clss.cpu()
                # print("clsss  ", clss_mod)
                
                # class collapse
                if len(clss_mod) > 0:
                    for i, cl in enumerate (clss_mod):
                        # change car if obj is 2 (car) or 5 (bus) or 6 (train) or 7 (truck) or 8 (boat)
                        if cl == 2 or cl == 5 or cl == 6 or cl == 7 or cl == 8:
                            # print("Before : ", clss_mod[i]) 
                            clss_mod = torch.tensor([2.])
                            # print("After : ", clss_mod[i])
                
                # print("clss_mod  ", clss_mod)
                
                
                # Write results
                if analyzer.show_yolo_result :
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        # label = None if detector.hide_labels else (detector.names[c] if detector.hide_conf else f'{detector.names[c]} "sdsd" {conf.cpu(): .2f}')
                        label = f'{detector.names[c]} {conf: .2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # pass detections to deepsort
                t4 = time_sync()
                # pass detections to Tracker (deepsort)
                deepsort_outputs = tracker.deepsort.update(xywhs.cpu(), confs_mod, clss_mod, im0)
                t5 = time_sync()
                # pass analysis to rdeepsort
                analyzer.update(deepsort_output=deepsort_outputs, confs=confs, names=detector.names, annotator=annotator)
                
            else:
                tracker.deepsort.increment_ages()
            
            # If want draw outer process but is unnecessary loop because can draw in process / use for develop debug
                
            for i in analyzer.buffer_list:
                buffer = analyzer.buffer_list[i]
                bbox = buffer.bbox
                c = int(buffer.class_id)  # integer class
                label = f'{buffer.track_id} {detector.names[c]} {buffer.conf:.2f} id : {c}'
                if analyzer.show_yolo_result is False :
                    annotator.box_label(bbox, label, color=colors(c, True))
                    
                if(buffer is not None and buffer.angle is not None):
                    cv.putText(im0, "angle : {:.2f}".format(buffer.angle), (bbox[0],bbox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                
                # draw line path of objective movement
                if((buffer.frame.get(1) is not None ) and buffer.frame.get(buffer.max_buffer) is not None):
                    x1, y1 = buffer.frame[1]
                    x2, y2 = buffer.frame[buffer.max_buffer]
                    # color = (0, 255, 0)
                    center_x, center_y = buffer.center
                    label = f'{buffer.track_id} '
                    box_draw = [
                        center_x - analyzer.length_check_polygon, 
                        center_y - analyzer.length_check_polygon, 
                        center_x + analyzer.length_check_polygon, 
                        center_y + analyzer.length_check_polygon
                    ]
                    annotator.box_label(box_draw, label, color=(100,255,0))
                    cv.line(im0, (x1,y1), (x2, y2), (0, 255, 0), 5)
                
                # draw path in polygon
                if(buffer.frame.get(buffer.max_buffer) is not None):
                    for idx in analyzer.polygon:
                        if analyzer.polygon[idx].track_ids.get(buffer.track_id) is not None:
                            x1, y1 = buffer.frame[1]
                            x2, y2 = buffer.frame[buffer.max_buffer]
                            center_x, center_y = buffer.center
                            label = f'{buffer.track_id} '
                            box_draw = [
                                center_x - analyzer.length_check_polygon, 
                                center_y - analyzer.length_check_polygon, 
                                center_x + analyzer.length_check_polygon, 
                                center_y + analyzer.length_check_polygon
                            ]
                            annotator.box_label(box_draw, label, color=(0,0,255))
                            # color = (0, 0, 255)
                            cv.line(im0, (x1,y1), (x2, y2), (0, 0, 255), 5)
            
            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
            
            # Stream results
            im0 = annotator.result()
            
            # delete clear buffer
            analyzer.delete_buffer()
            
            # show data in image
            fps = 1./(time.time()-time_start)
            cv.putText(im0, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            
            for i in analyzer.polygon:
                if i == 0:
                    cv.putText(im0, "COUNT CAR UP: {:.0f}".format(analyzer.polygon[i].count_up), (400,245), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv.putText(im0, "COUNT CAR DOWN: {:.0f}".format(analyzer.polygon[i].count_down), (400,265), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv.putText(im0, "COUNT CAR UNKNOW: {:.0f}".format(analyzer.polygon[i].count_unknow), (400,285), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                elif i == 1:
                    cv.putText(im0, "COUNT CAR UP: {:.0f}".format(analyzer.polygon[i].count_up), (1000,245), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv.putText(im0, "COUNT CAR DOWN: {:.0f}".format(analyzer.polygon[i].count_down), (1000,265), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv.putText(im0, "COUNT CAR UNKNOW: {:.0f}".format(analyzer.polygon[i].count_unknow), (1000,285), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                elif i == 2:
                    cv.putText(im0, "COUNT CAR UP: {:.0f}".format(analyzer.polygon[i].count_up), (1500,245), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv.putText(im0, "COUNT CAR DOWN: {:.0f}".format(analyzer.polygon[i].count_down), (1500,265), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv.putText(im0, "COUNT CAR UNKNOW: {:.0f}".format(analyzer.polygon[i].count_unknow), (1500,285), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                elif i == 3:
                    cv.putText(im0, "COUNT CAR UP: {:.0f}".format(analyzer.polygon[i].count_up), (1000,545), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv.putText(im0, "COUNT CAR DOWN: {:.0f}".format(analyzer.polygon[i].count_down), (1000,565), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv.putText(im0, "COUNT CAR UNKNOW: {:.0f}".format(analyzer.polygon[i].count_unknow), (1000,585), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    
            
        if show_vid:
            cv.imshow(p, im0)
            if cv.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
    
    print('Done. (%.3fs)' % (time.time() - t0))

                
    
    
if __name__ == "__main__":
    main()