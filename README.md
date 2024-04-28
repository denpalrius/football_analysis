# Football match analysis using YOLO v8 and OpenCV

This is a Computer Vision project usibg YOLO V8 and OpenCV in Python to analyse a Bundesliga football match.

- Train Yolo5 moidel with annotated dataset
- Use the model to predict and create the bounding boxes
- Track the bounding boxes 
    - Since we're not streaming we will pickle them to avoid costly inference
- Draw ellipse on the players and referee bounding boxes for better viewing experience
