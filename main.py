from utils import read_video, write_video
from trackers import Tracker, Annotator


def main():
    # Read the video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames=video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pkl')

    # Draw annotations
    annotator = Annotator()
    annotated_video_frames = annotator.draw_annotations(frames=video_frames,
                                                        tracks=tracks)

    # Save video
    write_video(annotated_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
