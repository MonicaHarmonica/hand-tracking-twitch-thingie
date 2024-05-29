import os
import time
import cv2
import numpy as np
import util
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw


deltaTime = 0

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

model_path = '\\'.join(os.path.realpath(__file__).split('\\')[:-1])+"\\hand_landmarker.task"

class Character:
    x = 13
    y = 0
    xspeed = 0
    yspeed = 0

    img = 0
    mask = 0

    w = 0
    h = 0

    is_held = True

    def __init__(self, img_path):
        img = cv2.imread(img_path)
        self.w = int(img.shape[1] / 5)
        self.h = int(img.shape[0] / 5)

        img = cv2.resize(img, (self.w, self.h))
        self.img = img
        ret, self.mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 256, cv2.THRESH_BINARY) 


    def move(self, destination):
        x = destination[0] - self.w / 2
        y = destination[1] - self.h / 3

        self.xspeed = (x - self.x)
        self.yspeed = (y - self.y)
        
    def update(self):

        dT = deltaTime * 10

        if not self.is_held:
            if self.yspeed < 300:
                self.yspeed += 9.8
        
        

        if self.x + self.xspeed< 0:
            self.xspeed = 0
            self.x = 0
        if self.y + self.yspeed < 0:
            self.yspeed = 0
            self.y = 0
        
        if self.x + self.xspeed * dT >= 800 - self.w:
            self.xspeed = 0
            self.x = 800 - self.w

        if self.y + self.yspeed * dT >= 600 - self.h:
            self.yspeed = 0
            self.y = 600 - self.h

        if self.y + 1 >= 600 - self.h:
            self.xspeed /= 2

        self.x += self.xspeed * dT
        self.y += self.yspeed * dT


    def draw(self, frame):

        xd = int(self.x)
        yd = int(self.y)

        roi = frame[yd:yd+self.h ,xd:xd+self.w]

        roi[np.where(self.mask)] = 0
        roi += self.img

def pinch_detect(frame, index, thumb, frame_size):
    
    (frame_w, frame_h) = frame_size

    dist = util.get_distance([index, thumb])
    if dist < 75:
        (ix, iy)  = index
        ix *= frame_w
        iy *= frame_h
        (tx, ty)  = thumb
        tx *= frame_w
        ty *= frame_h
        (x, y) = util.get_midpoint([(ix, iy), (tx, ty)])
        #cv2.putText(frame, "PINCH", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return (x, y)
    else:
        return None

prevTime = time.time()
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

charlie = Character("charlie.png")

was_pinching_last_frame = False

while cam.isOpened():
    deltaTime = time.time() - prevTime
    
    prevTime = time.time()

    succ,frame = cam.read()
    if not succ:
        break
    
    frame_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    processed = hands.process(frameRGB)

    landmark_list = []
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        mp_draw.draw_landmarks(frame, hand_landmarks)
        for lm in hand_landmarks.landmark:
            landmark_list.append((lm.x, lm.y))
        pinch_point = pinch_detect(frame, landmark_list[8], landmark_list[4], (frame_w, frame_h))
        if pinch_point != None:
            if util.point_in_rectangle(pinch_point, (charlie.x, charlie.y), (100, 100)):
                if was_pinching_last_frame == False:
                    charlie.is_held = True
            
            if charlie.is_held:
                charlie.move(pinch_point)
            
            was_pinching_last_frame = True
        else:
            charlie.is_held = False
            was_pinching_last_frame = False
    else:
        charlie.is_held = False
        was_pinching_last_frame = False

    charlie.update()
    charlie.draw(frame)
    

    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cam.release()
cv2.destroyAllWindows()