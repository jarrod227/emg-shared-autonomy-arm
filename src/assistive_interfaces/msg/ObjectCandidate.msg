# One closed-set object localized in the stereo reference frame.
#
# The track ID is temporary and is unique only within one tracker session.
# The position is a robust 3D reference point, not a complete grasp pose.

uint32 track_id
string class_label
float32 class_confidence
geometry_msgs/Point position
# Fraction of mask pixels retained as robust finite-depth inliers, in [0, 1].
float32 localization_confidence
