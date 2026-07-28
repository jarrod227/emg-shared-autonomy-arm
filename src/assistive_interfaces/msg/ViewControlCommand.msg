# Volatile, source-independent command for one bounded active-view axis.
#
# This is not general teleoperation. The handoff controller accepts it only in
# TARGET_SEARCH or HANDOFF_SEARCH. Activation selects a target angle around a
# configured safe center; it never selects robot speed.
#
# QoS contract: KEEP_LAST(1) + VOLATILE. Publishers must not retain or replay
# old commands.

uint8 HOLD=0
uint8 LEFT=1
uint8 RIGHT=2

std_msgs/Header header   # source time; freshness is judged on this stamp
uint8 direction          # HOLD, LEFT, or RIGHT
float32 activation       # session-normalized activation in [0, 1]
float32 confidence       # direction/activation confidence in [0, 1]
float32 signal_quality   # source signal quality in [0, 1]
uint32 sequence          # monotonically increasing per source/session
