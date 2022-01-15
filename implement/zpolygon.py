from shapely.geometry import Polygon

import cv2 as cv
import numpy as np
import math

class Zpolygon(object):
    
    def __init__(self, ps, color, id):
        # Define ps = [(x1, y1),(x2, y2)]
        self.id = id
        self.arr = np.array(ps)
        self.ps = ps
        self.color = color
        self.polygon = Polygon(self.ps)

        self.track_ids = {int: Datazpolygon}
        self.track_ids = {}
        
        self.count_up = 0
        self.count_down = 0
        self.count_unknow = 0

    

    def check_intersects(self, ps_check):
        # Define ps_check = [(x1, y1), (x2, y2)]
        polygon_check = Polygon(ps_check)
        
        isIntersection = self.polygon.intersects(polygon_check)
        return isIntersection
    
    
    def update(self, track_id, box, angle, poly_id=None):
        if self.track_ids.get(track_id) is None:
            print('###############')
            print("id : ", track_id, " enter polygon [{}]".format(poly_id))
            self.track_ids[track_id] = Datazpolygon(box=box, angle=angle)
            
        else:
            self.track_ids[track_id].save_path(box)
            
    def delete(self, track_id, poly_id=None):
        if self.track_ids.get(track_id) is not None:
            print("id : ", track_id, " out polygon [{}]".format(poly_id))
            print('###############')
            self.track_ids.pop(track_id)

    def draw_polygon(self, img):
        # Define ps = [(400,500),(600,200), (800, 200),(750,500),(600,580)]
        # print("POLY", ps)
        arr_ps = np.array(self.ps)
        cv.polylines(img, [arr_ps], True, self.color, thickness=3)
    
    def calculate_counting(self, track_id, box):
        if self.track_ids.get(track_id) is not None:
            center_x_current = box[0] + (abs(box[2] - box[0])//2)
            center_y_current = box[1] + (abs(box[3] - box[1])//2)
            center_x_first, center_y_first = self.track_ids[track_id].center
            print("id : {} ,center_f ({}, {})".format(track_id, center_x_first, center_y_first))
            angle_outer = self.track_ids[track_id].calculate_angle(center_x_first, center_y_first, center_x_current, center_y_current)
            if(self.track_ids[track_id].angle_enter is None):
                # define not analysis enter angle
                direct_normal = 30
            else:
                direct_normal = abs(angle_outer - self.track_ids[track_id].angle_enter)
                print("id: {} , angle_enter : ({})".format(track_id, self.track_ids[track_id].angle_enter))
            # show status   
            print("id: ", track_id, " x1y1 : ", "({:.2f}, {:.2f})".format(center_x_first, center_y_first), " x2y2 : ", "({:.2f}, {:.2f})".format(center_x_current, center_y_current))
            print("angle out polygon : ", angle_outer)
            print("direction normal polygon : ", direct_normal)
        
        # make sure when doesn't have object enter polygon    
        else:
            direct_normal = None
            angle_outer = None
        
        
        
        
        return direct_normal, angle_outer
    
    def show_value(self):
        print('track_ids : ', self.track_ids)



class Datazpolygon(object):
    def __init__(self, box, angle=None):
        self.box = box
        center_x = box[0] + (abs(box[2] - box[0])//2)
        center_y = box[1] + (abs(box[3] - box[1])//2)
        self.center = (center_x, center_y)
        # path is obj movement when obj enter polygon
        self.path = []
        self.angle_enter = angle
        self.angle_outer = None
    
    # Non use in current
    def save_path(self, box):
        self.path.append(box)
            
    def calculate_angle(self, xmin ,ymin, xmax, ymax):
        direct_x = xmax - xmin + 1
        direct_y = ymax - ymin + 1
        
        x_lenght = abs(xmax - xmin)
        y_lenght = abs(ymax - ymin)
        
        if(x_lenght) != 0:
            ini_angle = math.degrees(math.atan(y_lenght/x_lenght))
        else:
            ini_angle = 0
        
        base_angle = self.check_quadrant(direct_x, direct_y)
        
        angle = ini_angle + base_angle    
        self.angle_outer = angle
        return angle

    def check_quadrant(self, direct_x, direct_y):
        base_angle = 0
        # Quadrant 1
        if(direct_x > 0 and direct_y < 0):
            base_angle = 0
        
        # Quadrant 2
        if(direct_x < 0 and direct_y < 0):
            base_angle = 90
        
        # Quadrant 3
        if(direct_x < 0 and direct_y > 0):
            base_angle = 180
        
        # Quadrant 4
        if(direct_x > 0 and direct_y > 0):
            base_angle = 270
        
        return base_angle
        