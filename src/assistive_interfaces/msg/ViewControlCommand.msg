# Volatile, source-independent command for one bounded active-view axis.
#
# This is not general teleoperation. The handoff controller accepts it only in
# TARGET_SEARCH or HANDOFF_SEARCH. Activation selects a rate -- this used to
# promise the opposite, that an effort level named an absolute angle, and that
# promise was withdrawn because on a real wearer it made the arm retreat while
# they were pushing it outward.
#
# What still holds, and what actually makes the command safe to accept, is the
# bound: the reachable set is a band around a configured safe center, whatever
# sequence of commands arrives. Since a rate names no angle, the consumer is
# the only place that bound can live -- enforce it rather than trusting the
# sender, and stop the axis if commands stop arriving.
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
