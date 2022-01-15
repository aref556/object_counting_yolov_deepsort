
# Analyze is list of Bdeepsort
from numpy.core.defchararray import count
from implement.bdeepsort import Bdeepsort
import numpy as np
import sys


sys.path.insert(0, 'object_detection/yolov5')
from object_detection.yolov5.utils.plots import Annotator, colors

sys.path.insert(0, 'object_tracking')
from object_tracking.deep_sort.utils.parser import get_config

sys.path.insert(0, 'implement')
from implement.zpolygon import Zpolygon

# Define Polygon = [[(x1, y1),(x2, y2), ...], [(x1, y1),(x2, y2), ...]]

def bboxes_to_center(box):
    center_x = box[0] + (abs(box[2] - box[0])//2)
    center_y = box[1] + (abs(box[3] - box[1])//2)
    return center_x, center_y


class Analyzer(object):
    def __init__(self, path_config='implement/configs/analyzer.yaml', max_age=30, polygons=None):
        self.cfg = get_config()
        self.cfg.merge_from_file(path_config)
        # print(self.cfg.RDEEPSORT.MAX_CHANNEL)
        self.max_age = max_age
        self.max_channel = self.cfg.ANALYZER.MAX_CHANNEL
        self.min_length_update = self.cfg.ANALYZER.MIN_LENGTH_UPDATE_BUFFER
        #self.max_length_update = self.cfg.ANALYZER.MAX_LENGTH_UPDATE_BUFFER
        self.max_length_update = 5000
        self.show_yolo_result = self.cfg.ANALYZER.SHOW_YOLO_RESULT
        self.max_live_polygon_count = self.cfg.ANALYZER.MAX_LIVE_POLYGON_COUNT
        self.length_check_polygon = self.cfg.ANALYZER.LENGTH_CHECK_POLYGON
        self.polygons = self.cfg.ANALYZER.PLACE_TEST
        # print("OR POLYGONS: ", polygons)
        # print("OR TYPE : ", type(polygons))
        
        # print("POLUGON : ", self.polygons)
        # print("TYPE : ", type(self.polygons))
        
        
        # define parameter list of buffer deepsort ## type hint
        self.buffer_list = {int: Bdeepsort}
        self.buffer_list = {}
        
        # define Polygon Area
        self.polygon = {int: Zpolygon}
        self.polygon = {}
        
        # define counting parameter
        self.count_up = 0
        self.count_down = 0
        self.count_unknonw = 0
        
        if(self.polygons is not None):
            for i, polypoint in enumerate(self.polygons):
                if self.polygon.get(i) is None :
                    # Generate Object Polygon Polypoint = [(x1, y1), (x2, y2), ...]
                    if (i == 0):
                        color = (255, 255, 0)
                    elif (i == 1):
                        color = (0, 255, 0)
                    elif (i == 2):
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)
                    self.polygon[i] = Zpolygon(polypoint, color=color, id=i)

    
    def update(self, deepsort_output, confs, names, annotator: Annotator, img=None):
        if len(deepsort_output) > 0:
            for i, (output, conf) in enumerate(zip(deepsort_output, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]
            
                # center_current_x, center_current_y = bboxes_to_center(bboxes)   
                if (self.buffer_list.get(id) is None):
                    # create new buffer
                    self.buffer_list[id] = Bdeepsort(track_id=id, class_id=cls, bbox=bboxes, min_dist_update=self.min_length_update, max_buffer=5, conf=conf, max_dist_update=self.max_length_update)
                else:
                    # update main parameter buffer
                    
                    self.buffer_list[id].update(bboxes, conf)  
                    
                    ## Develop
                    # check in polygon
                    for i in self.polygon:
                        # # check for not analysis if object live in polygon
                        # if(self.buffer_list[id].in_polygon_flag is False): 
                        # polygon = self.polygon[i]
                        # box = [(bboxes[0], bboxes[1]), (bboxes[2], bboxes[1]), (bboxes[2], bboxes[3]), (bboxes[0], bboxes[3])]
                        center_x, center_y = bboxes_to_center(bboxes)
                        box = [
                            (center_x - self.length_check_polygon, center_y - self.length_check_polygon), 
                            (center_x + self.length_check_polygon, center_y - self.length_check_polygon), 
                            (center_x + self.length_check_polygon, center_y + self.length_check_polygon), 
                            (center_x - self.length_check_polygon, center_y + self.length_check_polygon)
                        ]
                        # label = f'{id} '
                        # box_draw = [center_x - 50, center_y - 50, center_x + 50, center_y + 50]
                        # annotator.box_label(box_draw, label, color=colors(cls, True))
                        isIntersect = self.polygon[i].check_intersects(box)
                        # print("isIntersect : ", isIntersect)
                        # self.buffer_list[id].update_enter_polygon_flag(isIntersect)
                        if(self.buffer_list.get(id) is not None):
                        # if(self.buffer_list.get(id) is not None and self.buffer_list[id].angle is not None):
                            if(isIntersect):
                                if(self.buffer_list[id].live_id is None):
                                    self.buffer_list[id].update_enter_polygon_flag(True)
                                    self.buffer_list[id].set_live_id(i)
                                    # print('ID : ',id, ' / enter polygon : ' , self.buffer_list[id].in_polygon_flag)
                                    self.polygon[i].update(track_id=id, box=bboxes, angle=self.buffer_list[id].angle, poly_id=i)
                                elif(self.buffer_list[id].live_id == i):
                                    # live in polygon
                                    self.polygon[i].update(track_id=id, box=bboxes, angle=self.buffer_list[id].angle, poly_id=i)
                                    # self.polygon[i].show_value()
                            else:
                                # check when buffer entered polygon and go out polygon
                                if(
                                    self.buffer_list[id].live_id is not None 
                                    and (self.buffer_list[id].live_id == i)
                                ):
                                    
                                    # set buffer enter polygon flag False because this buffer out of polygon
                                    self.buffer_list[id].update_enter_polygon_flag(False)
                                    # print('ID : ',id, ' / enter polygon : ' , self.buffer_list[id].in_polygon_flag)

                                    direct_normal, angle_outer = self.polygon[i].calculate_counting(id, bboxes)
                                    
                                    # New
                                    if(self.polygon[i].track_ids.get(id) is not None):
                                        count_live_polygon = len(self.polygon[i].track_ids[id].path)
                                        # if direction of obj move less than 90 is mean obj moving in the same direction
                                        if(
                                            direct_normal is not None 
                                            and direct_normal < 120
                                            and count_live_polygon >= self.max_live_polygon_count
                                            
                                        ):
                                            if(angle_outer > 180):
                                                self.polygon[i].count_down += 1
                                                print('id : {}, polygon [{}] count down'.format(id, i))
                                                self.polygon[i].delete(track_id=id, poly_id=i)
                                                self.buffer_list[id].live_id = None

                                            elif(angle_outer < 180):
                                                self.polygon[i].count_up += 1
                                                print('id : {}, polygon [{}] count up'.format(id, i))
                                                self.polygon[i].delete(track_id=id, poly_id=i)
                                                self.buffer_list[id].live_id = None

                                            else:
                                                print('id : {}, polygon [{}] count unknow 1 : angle out polygon is {:.2f}'.format(id, i, angle_outer))
                                                self.polygon[i].count_unknow += 1
                                                self.polygon[i].delete(track_id=id, poly_id=i)
                                                self.buffer_list[id].live_id = None 
                                        else:
                                            print('id : {}, polygon [{}] uncount : live polygon < value live polygon can config'.format(id, i))
                                            print("id : {}, live polygon : {}".format(id, count_live_polygon))
                                            print("id : {}, direct_normal : {:.2f}".format(id, direct_normal))
                                            # self.polygon[i].count_unknow += 1
                                            self.polygon[i].delete(track_id=id, poly_id=i)
                                            self.buffer_list[id].live_id = None
                                            
                                                 

                # draw in process avoid unnecessary loop
                # c = int(cls)  # integer class
                # label = f'{id} {names[c]} {conf:.2f}'
                # if self.show_yolo_result is False :
                #     annotator.box_label(bboxes, label, color=colors(c, True))
                
    
    def delete_buffer(self):
        mo = []
        for m in self.buffer_list:
            if(self.buffer_list[m].updated_flag is False):
                mo.append(m)
            else:
                self.buffer_list[m].updated_flag = False
        
        for t in mo:
            new_age = self.buffer_list[t].age + 1
            if new_age > self.max_age:
                if (self.buffer_list[t].in_polygon_flag):
                    print("***********************")
                    print("id : ", t, " enter but not out polygon")
                    
                    # set buffer enter polygon flag False because this buffer out of polygon
                    self.buffer_list[t].update_enter_polygon_flag(False)
            
                    direct_normal, angle_outer = self.polygon[self.buffer_list[t].live_id].calculate_counting(t, self.buffer_list[t].bbox)
                    if(self.polygon[self.buffer_list[t].live_id].track_ids.get(t) is not None):
                        count_live_polygon = len(self.polygon[self.buffer_list[t].live_id].track_ids[t].path)
    
                        # if direction of obj move less than 90 is mean obj moving in the same direction
                        if(
                            direct_normal is not None 
                            and direct_normal < 120
                            and count_live_polygon >= self.max_live_polygon_count
                        ):
                            if(angle_outer > 180):
                                self.polygon[self.buffer_list[t].live_id].count_down += 1
                                print('*** id : {}, polygon [{}] count down ***'.format(t, self.buffer_list[t].live_id))
                                self.polygon[self.buffer_list[t].live_id].delete(track_id=t, poly_id=self.buffer_list[t].live_id)
                                self.buffer_list[t].live_id = None
                            elif(angle_outer < 180):
                                self.polygon[self.buffer_list[t].live_id].count_up += 1
                                print('*** id : {}, polygon [{}] count up ***'.format(t, self.buffer_list[t].live_id))
                                self.polygon[self.buffer_list[t].live_id].delete(track_id=t, poly_id=self.buffer_list[t].live_id)
                                self.buffer_list[t].live_id = None
                            else:
                                print('*** id : {}, polygon [{}] count unknow 1 : angle out polygon is {:.2f} ***'.format(t, self.buffer_list[t].live_id, angle_outer))
                                self.polygon[self.buffer_list[t].live_id].count_unknow += 1
                                self.polygon[self.buffer_list[t].live_id].delete(track_id=t, poly_id=self.buffer_list[t].live_id) 
                                self.buffer_list[t].live_id = None
                        else:
                            print('*** id : {}, polygon [{}] uncount : live polygon < value live polygon can config ***'.format(t, self.buffer_list[t].live_id))
                            # self.polygon[self.buffer_list[t].live_id].count_unknow += 1
                            print("id : {}, live polygon : {}".format(t, count_live_polygon))
                            print("id : {}, direct_normal : {}".format(t, direct_normal))
                            self.polygon[self.buffer_list[t].live_id].delete(track_id=t, poly_id=self.buffer_list[t].live_id)
                            self.buffer_list[t].live_id = None
                # print("id : ", t, " deleted!")
                self.buffer_list.pop(t)
                # print(self.buffer_list.get(t))
            else:
                self.buffer_list[t].age = new_age
                # print("id : ", self.buffer_list[t].track_id, "miss", "/ age : ", self.buffer_list[t].age)
    



        
            
            
        
            