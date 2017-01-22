# Curved Lane Detection
An old, experimental implementation of curved lane detection using OpenCV, RANSAC, DBSCAN, perspective projection, some filters etc.
The resulted curves are in transformed space (a birdview of road, that could be possible to scale in cm) but it is displayed 
in argument reality by inverse transformation.

This is hardcoded (perspective transformation) for some random video, but not optimized for it. It is developed mostly for white lanes,
but sometimes can work for yellow as you can see in example video.
It runs real time on my machine.
It is quite robust (can estimate road center even if only one lane is visible, the lanes can be partially visible).

The algorithm is quite simple oldschool pipeline processing that would need to be improved and probably would perform worse than using DNN.

# Example 
The code effect can be seen in below video (click to watch):

[![Code ran for sample video](http://img.youtube.com/vi/H-GIlXoNKI0/0.jpg)](http://www.youtube.com/watch?v=H-GIlXoNKI0)
