#  VideoStablization

## Description

Motion-based algorithm for stablizing video. The algorithm works as follow

1. Detect features (strong corners) from video frame
2. Use Lucas-Kanade algorithm to track the displacement of features between two frames
3. Estimate transformation matrix from optical flow
4. Calculate a trajectory of displacement by cumulatively summing ttransformation matrics
5. Smooth the trajectory
6. Smooth the transformation matrix as $T_{Smoothed}(t) = T + (Trajectory_{Smoothed}(t) - Trajectory(t))$
7. Apply transformation matrix to video frame

## Algorithms

Two algorithms are implemented in video.cpp and streaming.cpp. They differs in the smoothing method. 
VIdeo.cpp applies moving average on trajectory, while streaming.cpp uses 1D Kalman filter. Accordingly, video.cpp need a complete video, but streaming.cpp does not.


## Notes
- This method relies on the performance of faeture detection and optical flow. Therefore, any condition that affects feature detection and optical flow, such as large displacement and sudden change of illuminance, may also lead to poor stabilization performance.
- This method cannot effectively deal with rolling shutter problem.

## Reference
Nghia Ho's [post](http://nghiaho.com/?p=2093).


