import cv2
import numpy as np
import time
from collections import deque, defaultdict
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
import torch
import os
from typing import Dict, List, Any, Callable, Optional

# Dependencies
try:
    from boxmot import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    log_message("❌ FaceNet (facenet-pytorch) NOT installed. Face Recognition DISABLED.", "WARNING")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

class FaceRecognitionManager:
    """Professional Face Recognition using FaceNet"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config['face_recognition']['enabled'] and FACENET_AVAILABLE
        self.similarity_threshold = config['face_recognition']['similarity_threshold']
        self.enrolled_faces = {}
        
        if self.enabled:
            try:
                self.face_detector = MTCNN(
                    keep_all=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    post_process=True
                )
                self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()
                if torch.cuda.is_available():
                    self.face_recognizer = self.face_recognizer.cuda()
                self._load_enrolled_faces()
            except Exception as e:
                log_message(f"Face Recognition Init Error: {e}", "ERROR")
                self.enabled = False

    def _load_enrolled_faces(self):
        enrolled_path = Path(self.config['face_recognition']['enrolled_faces_path'])
        if enrolled_path.exists():
            with open(enrolled_path, 'r') as f:
                data = json.load(f)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for key, value in data.items():
                candidate_id = value.get('name', key) if isinstance(value, dict) else key
                embedding_list = value.get('embedding') if isinstance(value, dict) else value
                if embedding_list:
                    self.enrolled_faces[candidate_id] = torch.tensor(embedding_list).to(device)
            log_message(f"✅ Loaded {len(self.enrolled_faces)} enrolled faces from {enrolled_path.absolute()}", "INFO")
        else:
            log_message(f"⚠️ Enrolled faces file NOT found at: {enrolled_path.absolute()}", "WARNING")
            log_message("Face Recognition will be DISABLED until a file is present.", "WARNING")
            self.enabled = False

    def identify_face(self, frame, bbox):
        if not self.enabled or not self.enrolled_faces:
            return None, 0.0
        try:
            x1, y1, x2, y2 = [max(0, int(coord)) for coord in bbox]
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: return None, 0.0
            
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(face_crop_rgb)
            if faces is None or len(faces) == 0: return None, 0.0
            
            face_tensor = faces[0].unsqueeze(0)
            if torch.cuda.is_available(): face_tensor = face_tensor.cuda()
            
            with torch.no_grad():
                detected_embedding = self.face_recognizer(face_tensor)
            
            best_match_id, best_similarity = None, 0.0
            for candidate_id, enrolled_embedding in self.enrolled_faces.items():
                similarity = torch.nn.functional.cosine_similarity(detected_embedding, enrolled_embedding.unsqueeze(0)).item()
                if similarity > best_similarity:
                    best_similarity, best_match_id = similarity, candidate_id
            
            if best_similarity >= self.similarity_threshold:
                return best_match_id, best_similarity
            return None, best_similarity
        except Exception as e:
            return None, 0.0

class MultiCandidateDetector:
    """Advanced Multi-Candidate Proctoring Logic (1300+ Line Parity)"""
    def __init__(self, config: Dict[str, Any], on_violation: Optional[Callable] = None):
        self.config = config
        self.on_violation = on_violation
        self.yolo = YOLO(config['detection']['yolo_model'])
        
        if BYTETRACK_AVAILABLE:
            self.tracker = ByteTrack(track_thresh=config['tracking']['track_thresh'], track_buffer=config['tracking']['track_buffer'], match_thresh=config['tracking']['match_thresh'], frame_rate=config['video']['process_fps'])
            self.object_tracker = ByteTrack(track_thresh=0.3, track_buffer=config['tracking']['track_buffer'], match_thresh=config['tracking']['match_thresh'], frame_rate=config['video']['process_fps'])
        
        self.face_landmarker = None
        if MEDIAPIPE_AVAILABLE:
            try:
                model_path = 'face_landmarker.task'
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True
                )
                self.face_landmarker = FaceLandmarker.create_from_options(options)
                log_message("✅ FaceLandmarker Model Loaded Successfully", "INFO")
            except Exception as e:
                log_message(f"❌ FaceLandmarker Init Failed: {e}", "ERROR")

        self.face_manager = FaceRecognitionManager(config)
        self.track_to_candidate_map = {}
        self.track_to_candidate_map = {}
        self.candidate_to_track_map = {}
        self.track_id_to_category = {} # Persistent object labels

        # 1080p Standard Seat Mapping (Dynamic Locking - Starts Empty)
        self.student_seats = {} 


        self.candidate_states = defaultdict(lambda: {
            'first_seen': None, 'last_seen': 0, 'face_history': defaultdict(float),
            'vote_count': 0, 'is_confirmed': False, 'hit_count': 0, 'risk_score': 0.0,
            'last_decay_time': time.time(), 'smoothed_bbox': None, 'absence_start': None
        })

        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

    def _get_head_pose(self, image_points, size):
        """Estimate head pose (Pitch, Yaw, Roll) using solvePnP"""
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1)) 
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success: return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        return euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]

    def _apply_risk_scoring(self, track_key: str, violation_type: str):
        """Increase candidate risk score and update session max"""
        state = self.candidate_states[track_key]
        points = self.config['thresholds'].get('points', {}).get(violation_type, 10)
        state['risk_score'] = min(100, state['risk_score'] + points)
        
        tid = track_key.split('_')[1]
        log_message(f"Scoring: Track {tid} +{points} pts ({violation_type}) -> Total: {state['risk_score']:.1f}", "INFO")
        return state['risk_score']

    def _apply_risk_decay(self, track_key: str, current_time: float):
        """Apply risk decay over time"""
        state = self.candidate_states[track_key]
        elapsed = current_time - state['last_decay_time']
        if elapsed >= 1.0:
            decay = self.config['thresholds'].get('decay_amount_per_sec', 5) * int(elapsed)
            state['risk_score'] = max(0.0, state['risk_score'] - decay)
            state['last_decay_time'] = current_time
        return state['risk_score']
    
    def _get_risk_band(self, score):
        """Map score to a named risk band"""
        bands = self.config['thresholds'].get('risk_bands', {'critical': 100, 'high': 20, 'medium': 12, 'low': 5})
        if score >= bands['critical']: return "CRITICAL"
        if score >= bands['high']: return "HIGH"
        if score >= bands['medium']: return "MEDIUM"
        if score >= bands['low']: return "LOW"
        return "CLEAR"

    def _check_gaze(self, landmarks, track_key: str, frame_time: float):
        """Check for iris gaze deviation using FaceMesh landmarks"""
        state = self.candidate_states[track_key]
        
        # Landmarks for gaze analysis
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]
        right_eye_outer = landmarks[263]
        right_eye_inner = landmarks[362]
        
        left_eye_top = landmarks[159]
        left_eye_bot = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bot = landmarks[374]

        left_width = abs(left_eye_inner.x - left_eye_outer.x) + 1e-6
        right_width = abs(right_eye_outer.x - right_eye_inner.x) + 1e-6
        left_height = abs(left_eye_bot.y - left_eye_top.y) + 1e-6
        right_height = abs(right_eye_bot.y - right_eye_top.y) + 1e-6

        left_gaze_ratio_x = (left_iris.x - left_eye_outer.x) / left_width
        right_gaze_ratio_x = (right_eye_outer.x - right_iris.x) / right_width
        avg_gaze_ratio_x = (left_gaze_ratio_x + right_gaze_ratio_x) / 2.0

        left_gaze_ratio_y = (left_iris.y - left_eye_top.y) / left_height
        right_gaze_ratio_y = (right_iris.y - right_eye_top.y) / right_height
        avg_gaze_ratio_y = (left_gaze_ratio_y + right_gaze_ratio_y) / 2.0
        
        t_left = self.config['thresholds'].get('gaze_threshold_left', 0.38)
        t_right = self.config['thresholds'].get('gaze_threshold_right', 0.68)
        t_top, t_bot = 0.35, 0.75
        
        is_deviating = (avg_gaze_ratio_x < t_left or avg_gaze_ratio_x > t_right or 
                        avg_gaze_ratio_y < t_top or avg_gaze_ratio_y > t_bot)
        
        if is_deviating:
            if 'gaze_start' not in state: state['gaze_start'] = frame_time
            elif frame_time - state['gaze_start'] > 8: # 8s sustained
                self._apply_risk_scoring(track_key, 'gaze_deviation')
                state['gaze_start'] = frame_time
                return True
        else:
            state.pop('gaze_start', None)
        return False

    def _check_head_orientation(self, landmarks, track_key: str, frame_time: float, frame_size):
        """Check for head orientation deviation using solvePnP"""
        state = self.candidate_states[track_key]
        h, w = frame_size[:2]
        image_points = []
        for idx in [1, 152, 263, 33, 291, 61]:
            lm = landmarks[idx]
            image_points.append([lm.x * w, lm.y * h])
        image_points = np.array(image_points, dtype="double")
        
        pitch, yaw, roll = self._get_head_pose(image_points, frame_size)
        
        is_deviating = abs(yaw) > 20 or abs(pitch) > 15
        if is_deviating:
            if 'head_start' not in state: state['head_start'] = frame_time
            elif frame_time - state['head_start'] > 5: # 5s sustained
                self._apply_risk_scoring(track_key, 'head_orientation')
                state['head_start'] = frame_time
                return True
        else:
            state.pop('head_start', None)
        return False

    def get_resolved_id(self, track_id: int, frame_time: float = 0):
        """Get stable name - Parity with proctor_test.py"""
        tid = int(track_id)
        # 1. Check permanent map (Locked after 5 successful votes)
        if tid in self.track_to_candidate_map:
            return self.track_to_candidate_map[tid]
        
        state = self.candidate_states.get(f"TRACK_{tid}")
        if not state: return f"TRACK_{tid}"
        
        # 2. Check Face History (Tentative Identification)
        if state['face_history']:
            best_match, total_score = max(state['face_history'].items(), key=lambda x: x[1])
            if best_match != 'UNKNOWN' and total_score > 0.4: # Lower threshold for parity
                return f"{best_match}?"
        
        # 3. Spatial Recovery (Seat-Locked)
        cx, cy = state.get('last_cx', 0), state.get('last_cy', 0)
        if cx > 0:
            for name, (sx, sy) in self.student_seats.items():
                dist = ((cx - sx)**2 + (cy - sy)**2)**0.5
                if dist < 200: 
                    return f"{name}?"
        
        return f"T{tid:02d}"

    def _check_proximity(self, candidates, frame_time):
        violations = []
        for i, c1 in enumerate(candidates):
            for i2, c2 in enumerate(candidates[i+1:]):
                if c1['track_id'] == c2['track_id']: continue
                b1, b2 = c1['bbox'], c2['bbox']
                is_overlap = not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])
                dist = np.linalg.norm(np.array(c1['center']) - np.array(c2['center']))
                avg_w = ((b1[2]-b1[0]) + (b2[2]-b2[0])) / 2
                if is_overlap or dist < (avg_w * 1.5):
                    cid1, cid2 = c1['id'], c2['id']
                    violations.append({
                        'type': 'proximity_violation', 'severity': 'MEDIUM', 'timestamp': frame_time,
                        'description': f"{cid1} and {cid2} sitting too close",
                        'candidate_id': f"{cid1}", # Primary (keep for backward compat)
                        'candidate_ids': [cid1, cid2], # NEW: Allow UI to show both names
                        'bboxes': [b1, b2], # Both parties in RED
                        'bbox': b1 # Fallback for single box viewers
                    })
        return violations

    def _check_absence(self, frame_time):
        """Check for physical occupancy in seat zones using ANY tracks"""
        violations = []
        if not self.student_seats: return violations
        
        occupied_seats = set()
        for track_key, state in self.candidate_states.items():
            if not track_key.startswith("TRACK_"): continue
            if frame_time - state['last_seen'] > 2.0: continue
            
            tx, ty = state.get('last_cx', 0), state.get('last_cy', 0)
            if tx == 0: continue
            
            for name, (sx, sy) in self.student_seats.items():
                dist = ((tx - sx)**2 + (ty - sy)**2)**0.5
                if dist < 250: # Seat proximity threshold
                    occupied_seats.add(name)

        for enrolled_name in self.face_manager.enrolled_faces.keys():
            person_key = f"PERSON_{enrolled_name}"
            if enrolled_name in occupied_seats:
                self.candidate_states[person_key]['last_seen'] = frame_time
                self.candidate_states[person_key]['absence_start'] = None
            else:
                last_valid = self.candidate_states.get(person_key, {}).get('last_seen', 0)
                if last_valid == 0:
                    self.candidate_states[person_key]['last_seen'] = frame_time
                    continue
                
                absence_duration = frame_time - last_valid
                if absence_duration > self.config['thresholds'].get('absence_seconds', 30):
                    if self.candidate_states[person_key]['absence_start'] is None:
                        self.candidate_states[person_key]['absence_start'] = frame_time
                        violations.append({
                            'type': 'candidate_absent', 'severity': 'HIGH', 'timestamp': frame_time,
                            'description': f"Seat of {enrolled_name} is empty for {int(absence_duration)}s",
                            'candidate_id': enrolled_name # Use raw name, not person_key
                        })
        return violations

    def process_frame(self, frame, frame_time: float):
        """Process single frame and detect violations (PORTAL SYNCED)"""
        h_orig, w_orig = frame.shape[:2]
        
        yolo_results = self.yolo.predict(
            frame, imgsz=1280, conf=self.config['detection']['confidence_threshold'],
            iou=self.config['detection']['iou_threshold'], verbose=False
        )[0]
        
        person_dets, object_dets = [], []
        prohib_ids = self.config['detection']['prohibited_class_ids']
        
        self.current_det_to_cat = {}
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf, bbox = float(box.conf[0]), box.xyxy[0].cpu().numpy()
            raw_name = yolo_results.names[cls_id]
            
            if cls_id == 0:
                # 1. Virtual Head Box: stabilize tracking on upper body/head
                w = bbox[2] - bbox[0]
                person_dets.append([bbox[0], bbox[1], bbox[2], bbox[1] + w*1.3, conf, cls_id])
            elif cls_id in prohib_ids and cls_id != 0: # STRICTLY ignore person class in object list
                object_dets.append(list(bbox) + [conf, cls_id])
                # Store mapping of detection index to class_id for later retrieval
                self.current_det_to_cat[len(object_dets)-1] = cls_id

        person_tracks = self.tracker.update(np.array(person_dets), frame) if person_dets else []
        object_tracks = self.object_tracker.update(np.array(object_dets), frame) if object_dets else []
        
        results = {'candidates': [], 'violations': []}
        active_persons = {}

        for track in person_tracks:
            tx1, ty1, tx2, ty2, track_id = track[:5]
            tid = int(track_id)
            track_key = f"TRACK_{tid}"
            state = self.candidate_states[track_key]
            state['last_seen'] = frame_time
            state['hit_count'] += 1
            state['last_cx'], state['last_cy'] = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            
            if state['first_seen'] is None: state['first_seen'] = frame_time
            if state['last_seen'] == 0: state['last_seen'] = frame_time

            # 2. EMA Smoothing
            if state['smoothed_bbox'] is None: state['smoothed_bbox'] = [tx1, ty1, tx2, ty2]
            else:
                s = state['smoothed_bbox']
                state['smoothed_bbox'] = [s[0]*0.7 + tx1*0.3, s[1]*0.7 + ty1*0.3, s[2]*0.7 + tx2*0.3, s[3]*0.7 + ty2*0.3]
            sx1, sy1, sx2, sy2 = state['smoothed_bbox']

            # Face Recog Consensus
            interval = self.config['tracking'].get('face_recognition_interval', 10)
            if not state['is_confirmed'] and state['hit_count'] % interval == 0:
                matched_id, sim = self.face_manager.identify_face(frame, [sx1, sy1, sx2, sy2])
                if matched_id:
                    state['face_history'][matched_id] += sim
                    state['vote_count'] += 1
                    
                    # Check Score-Based Lock (Parity with cctv_proctor_test.py)
                    # 3.75 total score is approx 5 votes @ 0.75 confidence
                    best_match, total_score = max(state['face_history'].items(), key=lambda x: x[1])
                    
                    if state['vote_count'] >= 5 or total_score > 3.75:
                        self.track_to_candidate_map[tid] = matched_id
                        self.candidate_to_track_map[matched_id] = tid
                        state['is_confirmed'] = True
                        
                        # DYNAMIC SEAT LOCKING: Learn seat position
                        self.student_seats[matched_id] = (state['last_cx'], state['last_cy'])

            cand_id = self.get_resolved_id(tid, frame_time)
            self._apply_risk_decay(track_key, frame_time)
            risk_band = self._get_risk_band(state['risk_score'])

            # 3. Behavioral Analysis
            landmarks = None
            # Reduced threshold from 80 to 40 to detect students in back rows
            if self.face_landmarker and (sy2-sy1) > 40: 
                crop = frame[max(0,int(sy1)):int(sy2), max(0,int(sx1)):int(sx2)]
                if crop.size > 0:
                    try:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        face_res = self.face_landmarker.detect_for_video(mp_image, int(frame_time * 1000))
                        if face_res.face_landmarks: landmarks = face_res.face_landmarks[0]
                    except: pass

            if landmarks:
                if self._check_gaze(landmarks, track_key, frame_time):
                    v_id = self.get_resolved_id(tid, frame_time)
                    results['violations'].append({
                        'type': 'gaze_deviation', 'severity': 'HIGH', 'timestamp': frame_time,
                        'description': f"Suspicious gaze for {v_id}", 'candidate_id': v_id,
                        'bbox': [int(sx1), int(sy1), int(sx2), int(sy2)]
                    })
                elif self._check_head_orientation(landmarks, track_key, frame_time, frame.shape):
                    v_id = self.get_resolved_id(tid, frame_time)
                    results['violations'].append({
                        'type': 'head_orientation', 'severity': 'HIGH', 'timestamp': frame_time,
                        'description': f"Suspicious head turn for {v_id}", 'candidate_id': v_id,
                        'bbox': [int(sx1), int(sy1), int(sx2), int(sy2)]
                    })

            candidate_data = {
                'id': cand_id, 'track_id': tid, 'bbox': [int(sx1), int(sy1), int(sx2), int(sy2)],
                'center': [state['last_cx'], state['last_cy']], 'risk_score': state['risk_score'], 'risk_band': risk_band
            }
            results['candidates'].append(candidate_data)
            active_persons[tid] = candidate_data

        # 4. Proximity & Absence
        results['violations'].extend(self._check_proximity(results['candidates'], frame_time))
        results['violations'].extend(self._check_absence(frame_time))

        # Handle Object Linkage
        for idx, o_track in enumerate(object_tracks):
            ox1, oy1, ox2, oy2, otid = o_track[:5]
            otid = int(otid)
            
            # Find nearest candidate logic (Restored)
            min_dist, nearest_tid = float('inf'), None
            ox, oy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            for tid, cand in active_persons.items():
                cx, cy = cand['center']
                dist = ((ox - cx)**2 + (oy - cy)**2)**0.5
                if dist < min_dist: min_dist, nearest_tid = dist, tid
            
            # Use Index 6 for Class ID if available (ByteTrack Standard), else ignore/fallback
            cls_id = int(o_track[6]) if len(o_track) > 6 else 67 # Fallback to Phone
            obj_name = self.yolo.names.get(cls_id, "Prohibited item")
            
            if nearest_tid and min_dist < 300:
                n_id = self.get_resolved_id(nearest_tid, frame_time)
                self._apply_risk_scoring(f"TRACK_{nearest_tid}", 'phone_detected')
                
                # STRICT PARITY: Red box on Person AND Object
                p_bbox = active_persons[nearest_tid]['bbox']
                o_bbox = [int(ox1), int(oy1), int(ox2), int(oy2)]
                
                results['violations'].append({
                    'type': f"prohibited_item_{str(obj_name).lower().replace(' ', '_')}", 'severity': 'CRITICAL', 'timestamp': frame_time,
                    'description': f"{obj_name} detected near {n_id}", 'candidate_id': n_id,
                    'bboxes': [p_bbox, o_bbox], # Both RED
                    'bbox': o_bbox # Fallback
                })

        # Final Callback
        if self.on_violation:
            for v in results['violations']:
                self.on_violation(v, frame)

        return results
