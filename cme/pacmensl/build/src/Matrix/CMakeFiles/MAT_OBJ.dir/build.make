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
include src/Matrix/CMakeFiles/MAT_OBJ.dir/depend.make

# Include the progress variables for this target.
include src/Matrix/CMakeFiles/MAT_OBJ.dir/progress.make

# Include the compile flags for this target's objects.
include src/Matrix/CMakeFiles/MAT_OBJ.dir/flags.make

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.o: src/Matrix/CMakeFiles/MAT_OBJ.dir/flags.make
src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.o: ../src/Matrix/FspMatrixBase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.o"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.o -c /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixBase.cpp

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.i"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixBase.cpp > CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.i

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.s"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixBase.cpp -o CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.s

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.o: src/Matrix/CMakeFiles/MAT_OBJ.dir/flags.make
src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.o: ../src/Matrix/FspMatrixConstrained.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.o"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.o -c /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixConstrained.cpp

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.i"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixConstrained.cpp > CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.i

src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.s"
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && /usr/local/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix/FspMatrixConstrained.cpp -o CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.s

MAT_OBJ: src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixBase.cpp.o
MAT_OBJ: src/Matrix/CMakeFiles/MAT_OBJ.dir/FspMatrixConstrained.cpp.o
MAT_OBJ: src/Matrix/CMakeFiles/MAT_OBJ.dir/build.make

.PHONY : MAT_OBJ

# Rule to build all files generated by this target.
src/Matrix/CMakeFiles/MAT_OBJ.dir/build: MAT_OBJ

.PHONY : src/Matrix/CMakeFiles/MAT_OBJ.dir/build

src/Matrix/CMakeFiles/MAT_OBJ.dir/clean:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix && $(CMAKE_COMMAND) -P CMakeFiles/MAT_OBJ.dir/cmake_clean.cmake
.PHONY : src/Matrix/CMakeFiles/MAT_OBJ.dir/clean

src/Matrix/CMakeFiles/MAT_OBJ.dir/depend:
	cd /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/Matrix /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix/CMakeFiles/MAT_OBJ.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/Matrix/CMakeFiles/MAT_OBJ.dir/depend

