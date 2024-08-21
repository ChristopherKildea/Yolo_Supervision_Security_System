import numpy as np
import supervision as sv
import cv2
from ultralytics import YOLO


model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()

# Path to the video you want to annotate
VIDEO_PATH = 'videos/thief_video3.mp4'


# Different polygon zones corresponding
# to different example videos
polygon = np.array([
    [51, 354],
    [203, 98],
    [407, 102],
    [495, 354],
    [51, 354]
])

# Polygon area for mall video
""""
polygon = np.array([
   [652, 1071],
   [960, 619],
   [1176, 615],
   [1752, 1075],
   [652, 1071]
])
"""

"""
# Polygon area for thief_video2.mp4
polygon = np.array([
    [1137, 374],
    [725, 238],
    [697, 258],
    [521, 190],
    [1, 418],
    [1, 514],
    [113, 714],
    [1013, 718],
    [1141, 370]
])
"""

"""
# Polygon area for thief_video.mp4
polygon = np.array([
    [247, 351],
    [304, 105],
    [393, 105],
    [516, 355],
    [247, 355]
])
"""

# Polygon zone of which we are detecting objects in
zone = sv.PolygonZone(polygon=polygon)

# initiate annotators
box_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)


# Number of photos taken
photo_count = 0
# Each person entering the frame is assigned an i.d.
captured_ids = set()


# Describes how to process each frame of the video
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    global photo_count

    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # 0 class i.d. is 'person'
    detections = tracker.update_with_detections(detections)
    # Sees if detections are in zone
    zone.trigger(detections=detections)

    # What we are labeling
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    for i in range(len(detections.xyxy)):
        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
        if tracker_id is not None and tracker_id not in captured_ids:
            captured_ids.add(tracker_id)
            photo_count += 1

            x1 = int(detections.xyxy[i][0])
            y1 = int(detections.xyxy[i][1])
            x2 = int(detections.xyxy[i][2])
            y2 = int(detections.xyxy[i][3])

            cropped_image = frame[y1:y2, x1:x2]

            # Save the frame with the new person detected
            photo_path = f"detected_persons/detected_person_{photo_count}.jpg"

            cv2.imwrite(photo_path, cropped_image)

    return frame


sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)

from IPython import display
display.clear_output()