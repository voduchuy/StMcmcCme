# Install script for directory: /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/tests

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
   "/usr/local/tests/test_fss")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_fss")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_fss" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_fss")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_fss")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_fss")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_mat")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_mat")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_mat" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_mat")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_mat")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_mat")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_ode")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_ode")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_ode" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_ode")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_ode")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_ode")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_fsp_solver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_fsp_solver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_fsp_solver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_fsp_solver")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_fsp_solver")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_fsp_solver")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_smfish")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_smfish")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_smfish" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_smfish")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_smfish")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_smfish")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_sensmat")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_sensmat")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_sensmat" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_sensmat")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_sensmat")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_sensmat")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_sensfsp_solver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_sensfsp_solver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_sensfsp_solver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_sensfsp_solver")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_sensfsp_solver")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_sensfsp_solver")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_petscwrap")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_petscwrap")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_petscwrap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_petscwrap")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_petscwrap")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_petscwrap")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/tests/test_stationaryfsp_solver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/tests" TYPE EXECUTABLE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/test_stationaryfsp_solver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/tests/test_stationaryfsp_solver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/tests/test_stationaryfsp_solver")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib"
      "$ENV{DESTDIR}/usr/local/tests/test_stationaryfsp_solver")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}/usr/local/tests/test_stationaryfsp_solver")
    endif()
  endif()
endif()

