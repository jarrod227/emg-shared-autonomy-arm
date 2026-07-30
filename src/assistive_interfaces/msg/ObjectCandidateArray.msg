# One timestamped stereo observation containing zero or more object candidates.
#
# header.stamp is the source-pair time and header.frame_id is the shared stereo
# reference frame. valid=true with an empty array is a fresh negative
# observation. valid=false means the input/synchronization/geometry was invalid
# and candidates must be empty.

std_msgs/Header header
bool valid
float32 pair_skew_sec
ObjectCandidate[] candidates
