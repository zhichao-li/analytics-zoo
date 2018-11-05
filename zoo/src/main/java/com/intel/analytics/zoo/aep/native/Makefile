CC       = g++
SUFFIX  ?= so

TARGET_DIR=./target
LIB  = $(TARGET_DIR)/libaep.$(SUFFIX)
SOURCES=./com_intel_analytics_zoo_aep_AEPHandler.cpp
OBJECTS=$(TARGET_DIR)/aep.o

all: $(LIB)

$(LIB): $(OBJECTS)
		$(CC) -Wall -shared $(OBJECTS) -lm -L /home/lizhichao/bin/god/memkind/ -l memkind -o $@

$(OBJECTS): $(SOURCES)
		mkdir -p $(TARGET_DIR)
		$(CC) -c -std=c++11 -I ${JAVA_HOME}/include -I ${JAVA_HOME}/include/linux -fPIC $<
		-o $@

clean:
		rm -rf $(OBJECTS) $(LIB)
