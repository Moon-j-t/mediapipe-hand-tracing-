import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, max_hands=2, detection_con=0.7, track_con=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # [수정됨] 손가락 걷기 상태를 디테일하게 추적하기 위한 딕셔너리
        self.walk_states = {
            'Left': {'prev_cx': None, 'prev_cy': None, 'prev_lead': None, 'active_frames': 0},
            'Right': {'prev_cx': None, 'prev_cy': None, 'prev_lead': None, 'active_frames': 0}
        }

    def find_hands(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def is_fist(self, hand_landmarks):
        tips = [8, 12, 16, 20] 
        pips = [6, 10, 14, 18] 
        wrist = hand_landmarks.landmark[0] 
        
        folded_count = 0
        for tip, pip in zip(tips, pips):
            tip_lm = hand_landmarks.landmark[tip]
            pip_lm = hand_landmarks.landmark[pip]
            dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
            dist_pip = math.hypot(pip_lm.x - wrist.x, pip_lm.y - wrist.y)
            if dist_tip < dist_pip:
                folded_count += 1
        return folded_count >= 3

    def is_walking_fingers(self, hand_landmarks, cx, cy, label):
        wrist = hand_landmarks.landmark[0]
        state = self.walk_states[label]
        
        # 1. 기본 자세 조건 (아래를 향하고 2손가락만 펴짐)
        if hand_landmarks.landmark[8].y < wrist.y or hand_landmarks.landmark[12].y < wrist.y:
            is_posture_correct = False
        else:
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            is_extended = []
            
            for tip, pip in zip(tips, pips):
                tip_lm = hand_landmarks.landmark[tip]
                pip_lm = hand_landmarks.landmark[pip]
                dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
                dist_pip = math.hypot(pip_lm.x - wrist.x, pip_lm.y - wrist.y)
                is_extended.append(dist_tip > dist_pip)
                
            index_middle_extended = is_extended[0] and is_extended[1]
            ring_pinky_folded = not is_extended[2] and not is_extended[3]
            is_posture_correct = index_middle_extended and ring_pinky_folded

        # 자세가 풀리면 해당 손의 상태 초기화 후 바로 False
        if not is_posture_correct:
            state['prev_cx'] = cx
            state['prev_cy'] = cy
            state['prev_lead'] = None
            state['active_frames'] = 0
            return False

        # 2. 이동 및 교차 판별
        is_horizontal_moving = False
        
        if state['prev_cx'] is not None:
            dx = cx - state['prev_cx']
            dy = cy - state['prev_cy']
            
            # [조건 1] 좌우(x축)로 2픽셀 이상 움직이고, 상하(y축) 움직임은 x축의 70% 미만일 때만 인정
            if abs(dx) > 2.0 and abs(dy) < (abs(dx) * 0.7):
                is_horizontal_moving = True

        # [조건 2] 검지와 중지의 상대적 위치(교차) 확인
        # 화면상 어느 손가락이 더 위/아래에 있는지 확인하여 걸음의 '교대'를 파악합니다.
        current_lead = hand_landmarks.landmark[8].y > hand_landmarks.landmark[12].y 

        if state['prev_lead'] is not None:
            if current_lead != state['prev_lead']:  # 손가락 위치가 역전(교차)되었다면
                # 교차 1회당 약 0.5초(15프레임) 동안 걷기 상태를 활성화(유지)합니다.
                state['active_frames'] = 15 

        # 다음 비교를 위해 현재 상태 저장
        state['prev_cx'] = cx
        state['prev_cy'] = cy
        state['prev_lead'] = current_lead
        
        if state['active_frames'] > 0:
            state['active_frames'] -= 1

        # 좌우로 이동 중이고 && 최근에 손가락이 교차했다면 최종 발동
        return is_horizontal_moving and (state['active_frames'] > 0)

    def get_hand_info(self, frame, results):
        hands_list = []
        current_labels = set() 

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label 
                current_labels.add(label)
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                
                hands_list.append({
                    "label": label,
                    "center": (cx, cy),
                    "landmarks": hand_landmarks,
                    "is_fist": self.is_fist(hand_landmarks),
                    "is_walking": self.is_walking_fingers(hand_landmarks, cx, cy, label) 
                })
        
        # 화면에서 손이 사라지면 오작동 방지를 위해 기록 초기화
        for key in self.walk_states.keys():
            if key not in current_labels:
                self.walk_states[key] = {'prev_cx': None, 'prev_cy': None, 'prev_lead': None, 'active_frames': 0}

        return hands_list