# Install script for directory: /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/tnfa_no_drug")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/tnfa_no_drug")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/tnfa_no_drug" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/tnfa_no_drug")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/tnfa_no_drug")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/tnfa_no_drug")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/signal_activated_bursting")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/signal_activated_bursting")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/signal_activated_bursting" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/signal_activated_bursting")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/signal_activated_bursting")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/signal_activated_bursting")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/hog1p")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/hog1p")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/hog1p" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/hog1p")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/hog1p")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/hog1p")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/time_invariant_bursting")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/time_invariant_bursting")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/time_invariant_bursting" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/time_invariant_bursting")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/time_invariant_bursting")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/time_invariant_bursting")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/repressilator")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/repressilator")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/repressilator" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/repressilator")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/repressilator")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/repressilator")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/transcr_reg_6d")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/transcr_reg_6d")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/transcr_reg_6d" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/transcr_reg_6d")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/transcr_reg_6d")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/transcr_reg_6d")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/hog1p_matvec")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/hog1p_matvec")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/hog1p_matvec" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/hog1p_matvec")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/hog1p_matvec")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/hog1p_matvec")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/examples/simple_petsc_program")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/examples" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/simple_petsc_program")
  if(EXISTS "$ENV{DESTDIR}/usr/local/examples/simple_petsc_program" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/examples/simple_petsc_program")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/examples/simple_petsc_program")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/examples/simple_petsc_program")
    endif()
  endif()
endif()

