# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/hog1p.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/hog1p.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/hog1p.dir/flags.make

examples/CMakeFiles/hog1p.dir/hog1p.cpp.o: examples/CMakeFiles/hog1p.dir/flags.make
examples/CMakeFiles/hog1p.dir/hog1p.cpp.o: ../examples/hog1p.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/hog1p.dir/hog1p.cpp.o"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples && /usr/local/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hog1p.dir/hog1p.cpp.o -c /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/examples/hog1p.cpp

examples/CMakeFiles/hog1p.dir/hog1p.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hog1p.dir/hog1p.cpp.i"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/examples/hog1p.cpp > CMakeFiles/hog1p.dir/hog1p.cpp.i

examples/CMakeFiles/hog1p.dir/hog1p.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hog1p.dir/hog1p.cpp.s"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/examples/hog1p.cpp -o CMakeFiles/hog1p.dir/hog1p.cpp.s

# Object files for target hog1p
hog1p_OBJECTS = \
"CMakeFiles/hog1p.dir/hog1p.cpp.o"

# External object files for target hog1p
hog1p_EXTERNAL_OBJECTS =

hog1p: examples/CMakeFiles/hog1p.dir/hog1p.cpp.o
hog1p: examples/CMakeFiles/hog1p.dir/build.make
hog1p: lib/libpacmensl.dylib
hog1p: /usr/local/petsc/lib/libpetsc.dylib
hog1p: /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libLAPACK.dylib
hog1p: /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib
hog1p: /opt/X11/lib/libX11.dylib
hog1p: /usr/local/lib/libparmetis.dylib
hog1p: /usr/local/lib/libmetis.dylib
hog1p: /usr/local/lib/libmpi.dylib
hog1p: /usr/lib/libc++.dylib
hog1p: /usr/lib/libdl.dylib
hog1p: examples/CMakeFiles/hog1p.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../hog1p"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hog1p.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/hog1p.dir/build: hog1p

.PHONY : examples/CMakeFiles/hog1p.dir/build

examples/CMakeFiles/hog1p.dir/clean:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/hog1p.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/hog1p.dir/clean

examples/CMakeFiles/hog1p.dir/depend:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/examples /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/examples/CMakeFiles/hog1p.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/hog1p.dir/depend

