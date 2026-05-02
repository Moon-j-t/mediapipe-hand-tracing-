import cv2
import mediapipe as mp
import math
import time  # 시간 측정을 위해 time 모듈 추가

class HandDetector:
    def __init__(self, max_hands=2, detection_con=0.7, track_con=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 걷기 상태 데이터에 'last_walk_time' (마지막 발동 시간) 추가
        self.walk_states = {
            'Left': {'start_x': None, 'start_y': None, 'prev_lead': None, 'cross_count': 0, 'last_walk_time': 0},
            'Right': {'start_x': None, 'start_y': None, 'prev_lead': None, 'cross_count': 0, 'last_walk_time': 0}
        }
        
        self.MIN_WALK_DIST = 40 

    def find_hands(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(img_rgb)

    def is_fist(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0] 
        tips = [4, 8, 12, 16, 20] 
        joints = [2, 6, 10, 14, 18] 
        
        for tip, joint in zip(tips, joints):
            tip_lm = hand_landmarks.landmark[tip]
            joint_lm = hand_landmarks.landmark[joint]
            
            dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
            dist_joint = math.hypot(joint_lm.x - wrist.x, joint_lm.y - wrist.y)
            
            # 엄지는 50%, 나머지는 15% 여유 허용 (이전 조건 유지)
            tolerance = 1.7 if tip == 4 else 1.15
            
            if dist_tip > dist_joint * tolerance:
                return False
                
        return True

    def is_walking_fingers(self, hand_landmarks, cx, cy, label):
        wrist = hand_landmarks.landmark[0]
        state = self.walk_states[label]
        
        # 1. 걷기 기본 자세 확인
        is_posture_correct = True
        if hand_landmarks.landmark[8].y < wrist.y or hand_landmarks.landmark[12].y < wrist.y:
            is_posture_correct = False
        else:
            tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
            is_extended = []
            for tip, pip in zip(tips, pips):
                dist_tip = math.hypot(hand_landmarks.landmark[tip].x - wrist.x, hand_landmarks.landmark[tip].y - wrist.y)
                dist_pip = math.hypot(hand_landmarks.landmark[pip].x - wrist.x, hand_landmarks.landmark[pip].y - wrist.y)
                is_extended.append(dist_tip > dist_pip)
            
            if not (is_extended[0] and is_extended[1] and not is_extended[2] and not is_extended[3]):
                is_posture_correct = False

        if not is_posture_correct:
            state['start_x'] = None
            state['cross_count'] = 0
            return False

        # 2. 크고 확실한 교차 감지
        y8 = hand_landmarks.landmark[8].y
        y12 = hand_landmarks.landmark[12].y
        margin = 0.02
        
        current_lead = state['prev_lead'] 
        
        if y8 - y12 > margin:
            current_lead = 'IndexDown'
        elif y12 - y8 > margin:
            current_lead = 'MiddleDown'
            
        if state['start_x'] is None:
            state['start_x'] = cx
            state['start_y'] = cy
            state['prev_lead'] = current_lead
            state['cross_count'] = 0
            return False

        if state['prev_lead'] is not None and current_lead != state['prev_lead']:
            state['cross_count'] += 1
            
        state['prev_lead'] = current_lead

        # 3. 기준점 대비 이동 검사
        total_dx = cx - state['start_x']
        total_dy = cy - state['start_y']

        is_correct_direction = (label == 'Left' and total_dx > 0) or (label == 'Right' and total_dx < 0)
        
        if not is_correct_direction:
            state['start_x'] = cx
            state['start_y'] = cy
            state['cross_count'] = 0
            return False

        is_horizontal_dominant = abs(total_dx) > 3 * abs(total_dy)

        if is_horizontal_dominant and state['cross_count'] >= 1 and abs(total_dx) > self.MIN_WALK_DIST:
            state['start_x'] = cx
            state['start_y'] = cy
            state['cross_count'] = 0
            return True

        return False

    def get_hand_info(self, frame, results):
        hands_list = []
        current_labels = set() 
        current_time = time.time()  # 현재 시간 측정

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label 
                current_labels.add(label)
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                
                # 먼저 걷기 이벤트를 판별합니다.
                is_walking = self.is_walking_fingers(hand_landmarks, cx, cy, label)
                
                # 걷기 이벤트가 발동했다면 현재 시간을 기록합니다.|
                if is_walking:
                    self.walk_states[label]['last_walk_time'] = current_time

                # [핵심] 주먹쥐기 판별: 마지막 걷기 발동 시간으로부터 0.5초가 안 지났으면 검사도 하지 않고 강제로 False 처리
                if current_time - self.walk_states[label]['last_walk_time'] < 0.5:
                    is_fist = False
                else:
                    is_fist = self.is_fist(hand_landmarks)
                
                hands_list.append({
                    "label": label,
                    "center": (cx, cy),
                    "landmarks": hand_landmarks,
                    "is_fist": is_fist,
                    "is_walking": is_walking 
                })
        
        for key in self.walk_states.keys():
            if key not in current_labels:
                self.walk_states[key] = {'start_x': None, 'start_y': None, 'prev_lead': None, 'cross_count': 0, 'last_walk_time': 0}

        return hands_list