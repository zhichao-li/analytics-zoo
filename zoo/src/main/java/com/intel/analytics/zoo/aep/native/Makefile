CC       = g++
SUFFIX  ?= so

TARGET_DIR=./target
LIB  = $(TARGET_DIR)/libape.$(SUFFIX)
SOURCES=./com_intel_analytics_zoo_aep_AEPSimulator.cpp
OBJECTS=$(TARGET_DIR)/aep.o

all: $(LIB)

$(LIB): $(OBJECTS)
		$(CC) -Wall -shared $(OBJECTS) -lm -o $@

$(OBJECTS): $(SOURCES)
		mkdir -p $(TARGET_DIR)
		$(CC) -c -I ${JAVA_HOME}/include -I ${JAVA_HOME}/include/linux -fPIC $< -o $@

clean:
		rm -rf $(OBJECTS) $(EXECUTABLE)
