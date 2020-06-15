# Install script for directory: /Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/OdeSolver

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/PACMENSL" TYPE FILE FILES
    "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/OdeSolver/OdeSolverBase.h"
    "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/OdeSolver/CvodeFsp.h"
    "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/OdeSolver/KrylovFsp.h"
    "/Users/huyvo/Codes/Projects/Computing/StMcmcCme/cme/pacmensl/src/OdeSolver/TsFsp.h"
    )
endif()

