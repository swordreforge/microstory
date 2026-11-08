CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -pthread
LDFLAGS ?= -pthread
TARGET := stories

$(TARGET): stories.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
