# Install script for directory: /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/lib/libpacmensl.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpacmensl.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpacmensl.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpacmensl.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/PACMENSL" TYPE FILE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/pacmensl_all.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/pacmensl.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Sys/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Models/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Partitioner/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/StateSet/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Matrix/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/OdeSolver/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/Fsp/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/SmFish/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/SensFsp/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/PetscWrap/cmake_install.cmake")
  include("/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/build/src/StationaryFsp/cmake_install.cmake")

endif()

