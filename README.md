# Custom security system with Yolov8, ByteTrack, and Supervision

This repository showcases a sample security system that monitors video/camera footage. Users define a zone of interest within said footage. When a person enters one of these zones, they are automatically detected, annotated, and tracked throughout their presence. To maintain a record, the system captures a photo of each individual upon their first entry into the zone.

Yolov8 is employed for object detection, and ByteTrack tracks the ‘person’ object once it has entered the zone. Supervision is used for annotating.




[Watch the video demonstration](ResultVideoExample.mp4)
