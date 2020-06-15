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
include tests/CMakeFiles/test_fsp_solver.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_fsp_solver.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_fsp_solver.dir/flags.make

tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o: tests/CMakeFiles/test_fsp_solver.dir/flags.make
tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o: ../tests/test_fsp_solver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests && /usr/local/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o -c /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/tests/test_fsp_solver.cpp

tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.i"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/tests/test_fsp_solver.cpp > CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.i

tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.s"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/tests/test_fsp_solver.cpp -o CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.s

# Object files for target test_fsp_solver
test_fsp_solver_OBJECTS = \
"CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o"

# External object files for target test_fsp_solver
test_fsp_solver_EXTERNAL_OBJECTS =

test_fsp_solver: tests/CMakeFiles/test_fsp_solver.dir/test_fsp_solver.cpp.o
test_fsp_solver: tests/CMakeFiles/test_fsp_solver.dir/build.make
test_fsp_solver: lib/libpacmensl.dylib
test_fsp_solver: /usr/local/lib/libgtest.dylib
test_fsp_solver: /usr/local/lib/libgtest_main.dylib
test_fsp_solver: /usr/local/petsc/lib/libpetsc.dylib
test_fsp_solver: /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libLAPACK.dylib
test_fsp_solver: /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib
test_fsp_solver: /opt/X11/lib/libX11.dylib
test_fsp_solver: /usr/local/lib/libparmetis.dylib
test_fsp_solver: /usr/local/lib/libmetis.dylib
test_fsp_solver: /usr/local/lib/libmpi.dylib
test_fsp_solver: /usr/lib/libc++.dylib
test_fsp_solver: /usr/lib/libdl.dylib
test_fsp_solver: tests/CMakeFiles/test_fsp_solver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../test_fsp_solver"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_fsp_solver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_fsp_solver.dir/build: test_fsp_solver

.PHONY : tests/CMakeFiles/test_fsp_solver.dir/build

tests/CMakeFiles/test_fsp_solver.dir/clean:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_fsp_solver.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_fsp_solver.dir/clean

tests/CMakeFiles/test_fsp_solver.dir/depend:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/tests /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tests/CMakeFiles/test_fsp_solver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_fsp_solver.dir/depend

