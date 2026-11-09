CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -pthread
LDFLAGS ?= -pthread
LDLIBS ?= -luring
TARGET := stories_server

$(TARGET): stories_server.cpp ../cpp/stories.cpp
	$(CXX) $(CXXFLAGS) -o $@ stories_server.cpp $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
