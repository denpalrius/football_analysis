import os
from ultralytics import YOLO
import supervision as sv

from utils import load_stub, save_stub

tracks = {
    "players" : [],
    "referees" : [],
    "ball" : []
}


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20 # Helps with better memory management. Will predict just 20 images at once
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
                    
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):
    
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("Returning pickled tracks!")
            print(load_stub(stub_path))
            return load_stub(stub_path)
        
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = { v:k for k, v in cls_names.items() }

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goal keeper to player since we are not treating the goal keeper in any special way for now
            for object_idx, cls_id in enumerate(detection_supervision.class_id):
                if(cls_names[cls_id] == "goalkeeper"):
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Create an object to organise the classes and their tracked bounding boxes
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # The ball is not tracked, so we will just save the last detection of the ball
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        save_stub(tracks, stub_path);
        print("Saved tracks stub pickeled successfully!")

        return tracks  

    def draw_annotations(self, frames, tracks):
        putput_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            player_dict = tracks["players"]
            referee_dict = tracks["referees"]
            ball_dict = tracks["ball"]

            # Draw players ellipses for bouding boxes to make the viewing experience better
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], track_id, color = (0, 255, 0) )


        
        return putput_video_frames

    def draw_ellipse(self, frame, bbox, track_id, color = (0, 255, 0)):
        x, y, w, h = bbox
        