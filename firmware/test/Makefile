# Host-side tests for the firmware sources that have no HAL dependency.
#
# Deliberately plain gcc, not the ARM toolchain: emg_packet.c and emg_filter.c
# are pure logic, so they can be tested on a workstation long before the CubeMX
# project exists or the board is flashed. They are the same files that get
# compiled into the firmware image -- one source, two compilers.
#
#   make check   build, run every test, and regenerate the cross-check fixtures
#   make clean

CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O1 -I../src
LDLIBS_FILTER := -lm

PACKET_BINARY := test_emg_packet
FILTER_BINARY := test_emg_filter
FIXTURES := fixture.bin golden.bin

.PHONY: check clean

check: $(PACKET_BINARY) $(FILTER_BINARY)
	./$(PACKET_BINARY)
	./$(PACKET_BINARY) --emit fixture.bin
	./$(FILTER_BINARY)
	./$(FILTER_BINARY) --emit golden.bin

$(PACKET_BINARY): test_emg_packet.c ../src/emg_packet.c ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_packet.c ../src/emg_packet.c

$(FILTER_BINARY): test_emg_filter.c ../src/emg_filter.c ../src/emg_filter.h
	$(CC) $(CFLAGS) -o $@ test_emg_filter.c ../src/emg_filter.c $(LDLIBS_FILTER)

clean:
	rm -f $(PACKET_BINARY) $(FILTER_BINARY) $(FIXTURES)
