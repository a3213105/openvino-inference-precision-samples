OPENVINO_INCLUDE ?= /home/sgui/miniforge3/lib/python3.12/site-packages/openvino/include
OPENVINO_LIB ?= /home/sgui/miniforge3/lib
PYTHON ?= /home/sgui/miniforge3/bin/python

MODEL ?= /home/sgui/Didi/old_docs/cpu_release_memory_api_prediction_test/ov_models/mlplanner_bs1/mlplanner_bs1.xml
PRECISION ?= bf16

CC ?= gcc
CXX ?= g++
CFLAGS ?= -O2 -Wall -Wextra
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17
LDFLAGS ?=

C_BIN := sample_c
CPP_BIN := sample_cpp
RPATH := -Wl,-rpath,$(OPENVINO_LIB)

.PHONY: all c cpp run-c run-cpp run-py clean

all: c cpp

c: $(C_BIN)

cpp: $(CPP_BIN)

$(C_BIN): sample_inference_precision.c
	$(CC) $(CFLAGS) -I$(OPENVINO_INCLUDE) $< $(OPENVINO_LIB)/libopenvino_c.so.2500 $(RPATH) $(LDFLAGS) -o $@

$(CPP_BIN): sample_inference_precision.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENVINO_INCLUDE) $< $(OPENVINO_LIB)/libopenvino.so.2500 $(RPATH) $(LDFLAGS) -o $@

run-c: c
	./$(C_BIN) $(MODEL) $(PRECISION)

run-cpp: cpp
	./$(CPP_BIN) $(MODEL) $(PRECISION)

run-py:
	$(PYTHON) sample_inference_precision.py $(MODEL) $(PRECISION)

clean:
	rm -f $(C_BIN) $(CPP_BIN)
