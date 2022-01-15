import numpy as np
import math

## Bdeepsort is buffer for get attribuite from deepsort for develop
class Bdeepsort(object):
    def __init__(self, track_id, class_id, bbox, min_dist_update, max_buffer, conf, max_dist_update):
        self.track_id = track_id
        self.class_id = class_id
        self.bbox = bbox
        self.min_dist_update = min_dist_update
        self.max_dist_update = max_dist_update
        self.max_buffer = max_buffer
        self.conf = conf
        self.before_center = None
        
        self.frame = {}
        self.count_frame = 0
        self.angle = None
        self.age = 0
        self.updated_flag = True
        self.in_polygon_flag = False
        self.live_id = None
        self.center = self.bboxes_to_center(bbox)
        
    def bboxes_to_center(self, box):
        center_x = box[0] + (abs(box[2] - box[0])//2)
        center_y = box[1] + (abs(box[3] - box[1])//2)
        return center_x, center_y
    
    def calculate_distance_p2p(self, p1, p2):
        a = np.array((p1))
        b = np.array((p2))
        return np.linalg.norm(a-b)
    
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
    
    # update paremeter tracking
    def update(self, box, conf):
        self.updated_flag = True
        x_curr, y_curr = self.bboxes_to_center(box)
        self.center = (x_curr, y_curr)
        self.bbox = box
        self.conf = conf
        name_frame = self.count_frame + 1
        name_before_frame = self.count_frame
        if(self.count_frame < self.max_buffer):
            # if(self.frame.get(name_frame) is None):
            if (self.frame.get(name_before_frame) is not None):
                distance = self.calculate_distance_p2p((self.frame[name_before_frame]), (x_curr, y_curr))
                #if (distance > self.min_dist_update and distance < self.max_dist_update):
                if (distance > self.min_dist_update and distance < self.max_dist_update):
                    self.frame[name_frame] = (x_curr, y_curr)
                    self.count_frame += 1     
            else:
                self.frame[name_frame] = (x_curr, y_curr)
                self.count_frame += 1
            # else:
                
            #     # if (distance < self.max_dist_update):
            #     self.frame.update({name_frame: (x_curr, y_curr)})
            #     self.count_frame += 1
                
        else:
            # check to make sure object moving
            before_last_frame = self.max_buffer - 1 
            distance = self.calculate_distance_p2p((self.frame[before_last_frame]), (x_curr, y_curr))
            # print(distance)
            
            #if (distance > self.min_dist_update and distance < self.max_dist_update):
            if (distance > self.min_dist_update):
                # shift frame like window
                for i in range(0, self.max_buffer):
                    name_update = i+1
                    name_dest = i+2
                    if((i+1) == self.max_buffer):
                        self.frame.update({name_update: (x_curr, y_curr)})
                    else:
                        self.frame.update({name_update: self.frame[name_dest]})
                
            # and then calculate angle 
            x1, y1 = self.frame[1]
            x2, y2 = self.frame[self.max_buffer]
            
            angle = self.calculate_angle(x1, y1, x2, y2)
            self.angle = angle
    
    def update_enter_polygon_flag(self, flag):
        self.in_polygon_flag = flag   
    
    def set_live_id(self, id):
        self.live_id = id