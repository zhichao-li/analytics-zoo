CC       = g++
SUFFIX  ?= so

TARGET_DIR=../../../../../../../../resources/native/
LIB  = $(TARGET_DIR)/libpersistent_memory_allocator.$(SUFFIX)
SOURCES=./com_intel_analytics_zoo_persistent_memory_PersistentMemoryAllocator.cpp
OBJECTS=$(TARGET_DIR)/persistent_memory_allocator.o

all: $(LIB)

$(LIB): $(OBJECTS)
		$(CC) -Wall -shared $(OBJECTS) -lm  -lmemkind -o $@

$(OBJECTS): $(SOURCES)
		mkdir -p $(TARGET_DIR)
		$(CC) -c -std=c++11 -I ${JAVA_HOME}/include -I ${JAVA_HOME}/include/linux -fPIC $< -o $@
driver: driver.cpp
		$(CC) -std=gnu++11 -I /usr/local/include driver.cpp -lmemkind -o $(TARGET_DIR)/driver.o

clean:
		rm -rf $(TARGET_DIR)

# cd zoo/target/classes
# javah -d zoo/src/main/java/com/intel/analytics/zoo/aep com.intel.analytics.zoo.persistent.memory.PersistentMemoryAllocator