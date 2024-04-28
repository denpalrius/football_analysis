import cv2
import numpy as np
from utils import get_bbox_center, get_bbox_width


class Annotator:

    def draw_annotations(self, frames, tracks):
        output_video_frames = []

        red = (0, 0, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player and referee ellipses for bouding boxes to make the viewing experience better

            # Draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], color=red)
                frame = self.draw_text_rect(
                    frame, player["bbox"], track_id, color=red)

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], color=yellow)

            # Draw ball triangle for ball's bouding box
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], color=green)

            output_video_frames.append(frame)

        print("Finished drawing annotations")

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color=(0, 0, 255)):
        _, _, _, y2 = bbox
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(img=frame,
                    center=(x_center, int(y2)),
                    axes=(width, int(0.35 * width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)

        return frame

    def draw_text_rect(self, frame, bbox, track_id, color=(0, 0, 255)):
        _, _, _, y2 = bbox
        x_center, _ = get_bbox_center(bbox)

        # Draw rectangle for players
        rec_width = 40
        rec_height = 20

        x1_rect = int(x_center - rec_width // 2)
        x2_rect = int(x_center + rec_width // 2)
        y1_rect = int((y2 - rec_height//2) + 15)
        y2_rect = int((y2 + rec_height//2) + 15)

        cv2.rectangle(img=frame,
                      pt1=(x1_rect, y1_rect),
                      pt2=(x2_rect, y2_rect),
                      thickness=cv2.FILLED,
                      color=color)

        x1_text = x1_rect + 12
        y1_text = y1_rect + 15

        # Move the text to the left if the track_id is too big
        if track_id > 99:
            x1_text -= 12

        cv2.putText(img=frame,
                    text=str(track_id),
                    org=(x1_text, y1_text),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 0),
                    thickness=2)

        return frame

    def draw_triangle(self, frame, bbox, color=(255, 0, 0)):
        x, _ = get_bbox_center(bbox)
        y = int(bbox[1])

        triangle_points = np.array([[x, y],
                                    [x - 10, y - 20],
                                    [x + 10, y - 20]])

        triangle = np.array(triangle_points, np.int32)

        # Draw the triangle over the ball
        cv2.drawContours(image=frame,
                         contours=[triangle],
                         contourIdx=0,
                         color=color,
                         thickness=cv2.FILLED)

        # Draw the triangle border
        cv2.drawContours(image=frame,
                         contours=[triangle],
                         contourIdx=0,
                         color=(0, 0, 0),
                         thickness=2)

        return frame
