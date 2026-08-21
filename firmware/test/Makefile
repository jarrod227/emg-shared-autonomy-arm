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
FEATURES_BINARY := test_emg_features
CLASSIFIER_BINARY := test_emg_classifier
GATE_BINARY := test_emg_gate
ACTIVATION_BINARY := test_emg_activation
RX_BINARY := test_emg_rx
VIEW_BINARY := test_emg_view
BINARIES := $(PACKET_BINARY) $(FILTER_BINARY) $(FEATURES_BINARY) \
            $(CLASSIFIER_BINARY) $(GATE_BINARY) $(ACTIVATION_BINARY) \
            $(RX_BINARY) $(VIEW_BINARY)
FIXTURES := fixture.bin golden.bin features.bin classifier.bin gate.bin \
            activation.bin view.bin

.PHONY: check clean

check: $(BINARIES)
	./$(PACKET_BINARY)
	./$(PACKET_BINARY) --emit fixture.bin
	./$(FILTER_BINARY)
	./$(FILTER_BINARY) --emit golden.bin
	./$(FEATURES_BINARY)
	./$(FEATURES_BINARY) --emit features.bin
	./$(CLASSIFIER_BINARY)
	./$(CLASSIFIER_BINARY) --emit classifier.bin
	./$(GATE_BINARY)
	./$(GATE_BINARY) --emit gate.bin
	./$(ACTIVATION_BINARY)
	./$(ACTIVATION_BINARY) --emit activation.bin
	./$(RX_BINARY)
	./$(VIEW_BINARY)
	./$(VIEW_BINARY) --emit view.bin

$(PACKET_BINARY): test_emg_packet.c ../src/emg_packet.c ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_packet.c ../src/emg_packet.c

$(FILTER_BINARY): test_emg_filter.c ../src/emg_filter.c ../src/emg_filter.h
	$(CC) $(CFLAGS) -o $@ test_emg_filter.c ../src/emg_filter.c $(LDLIBS_FILTER)

$(FEATURES_BINARY): test_emg_features.c ../src/emg_features.c ../src/emg_features.h
	$(CC) $(CFLAGS) -o $@ test_emg_features.c ../src/emg_features.c $(LDLIBS_FILTER)

$(CLASSIFIER_BINARY): test_emg_classifier.c ../src/emg_classifier.c \
                       ../src/emg_classifier.h ../src/emg_classifier_model.h \
                       ../src/emg_features.h ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_classifier.c ../src/emg_classifier.c

$(GATE_BINARY): test_emg_gate.c ../src/emg_gate.c ../src/emg_gate.h \
                 ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_gate.c ../src/emg_gate.c

$(ACTIVATION_BINARY): test_emg_activation.c ../src/emg_activation.c \
                       ../src/emg_activation.h ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_activation.c ../src/emg_activation.c

$(RX_BINARY): test_emg_rx.c ../src/emg_rx.c ../src/emg_rx.h \
               ../src/emg_packet.c ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_rx.c ../src/emg_rx.c ../src/emg_packet.c

clean:
	rm -f $(BINARIES) $(FIXTURES)

$(VIEW_BINARY): test_emg_view.c ../src/emg_view.c ../src/emg_view.h \
                ../src/emg_packet.h
	$(CC) $(CFLAGS) -o $@ test_emg_view.c ../src/emg_view.c
