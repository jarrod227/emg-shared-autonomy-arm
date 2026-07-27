# Discrete user intent from the biosignal (EMG) channel, or a simulated source.
#
# Contract (see docs/system_design.md): this stream is RELIABLE + VOLATILE.
# Publishers must NOT use TRANSIENT_LOCAL / retain / replay — a stale ABORT
# replayed to a late subscriber would spuriously cancel motion. REST is the
# classifier's idle class and is NOT published (it produces no event); the
# constant is listed only to document the full label set.

uint8 REST=0
uint8 NEXT_TARGET=1
uint8 CONFIRM=2
uint8 ABORT=3

std_msgs/Header header   # source time (stamp) — freshness is judged on this,
                         # never on message receipt time
uint8 command            # one of the constants above (REST never sent)
float32 confidence       # classifier confidence in [0, 1]
uint32 sequence          # monotonically increasing per source; gap = dropped
