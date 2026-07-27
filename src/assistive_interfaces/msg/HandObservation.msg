# 3D hand-presence cue for the handoff delivery-volume gate.
#
# In Objective 4.1 a simulated publisher fills this in; in Objective 4.2 it is
# produced by stereo triangulation of hand keypoints. The handoff controller
# treats a missing, stale, or invalid observation as "no hand" and fails closed
# (release is blocked). This cue is NOT safety-rated separation monitoring.

std_msgs/Header header       # source time (stamp) + reference frame; freshness
                             # is judged on stamp, never on receipt time
bool valid                   # true only when a hand was actually localized;
                             # false = explicit no-hand (still a valid message)
geometry_msgs/Point point    # 3D hand/palm position in header.frame_id (valid only if valid)
float32 confidence           # detection/triangulation confidence in [0, 1]

# Stereo-quality fields — populated in Objective 4.2; left 0.0 by the
# Objective 4.1 simulated publisher.
float32 pair_skew_sec        # timestamp skew between the left/right image pair
float32 reprojection_error   # triangulation reprojection error (pixels)
