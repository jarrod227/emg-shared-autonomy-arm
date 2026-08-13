# Host-side tests for the sEMG wire encoder.
#
# Deliberately plain gcc, not the ARM toolchain: emg_packet.c has no HAL or
# CMSIS dependency, so it can be tested on a workstation long before the
# CubeMX project exists or the board is flashed.
#
#   make check   build, run the tests, and regenerate fixture.bin
#   make clean

CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O1 -I../src

SOURCES := test_emg_packet.c ../src/emg_packet.c
BINARY := test_emg_packet
FIXTURE := fixture.bin

.PHONY: check clean

check: $(BINARY)
	./$(BINARY)
	./$(BINARY) --emit $(FIXTURE)

$(BINARY): $(SOURCES) ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(BINARY) $(FIXTURE)
