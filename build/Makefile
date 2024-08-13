# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -Wextra -std=c++11

# Directories
SRCDIR = ../norch/csrc
BUILDDIR = ../build
TARGET = ../norch/libtensor.so

# Files
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) --shared -o $(TARGET) $(OBJS)

# Rule to compile C++ source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Clean rule
clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)
