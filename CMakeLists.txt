# TWT Controller Python Bindings Module CMakeLists.txt
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# Lab: SHINE Lab, Texas State University
# PI: Marcelo Menezes De Carvalho
# This module provides Python-C++ communication interface for TWT Controller

cmake_minimum_required(VERSION 3.10)

# Project information
set(PROJECT_AUTHOR "Ahmed Maksud")
set(PROJECT_EMAIL "ahmed.maksud@email.ucr.edu")
set(PROJECT_LAB "SHINE Lab, Texas State University")
set(PROJECT_PI "Marcelo Menezes De Carvalho")
set(PROJECT_DESCRIPTION "NS3-AI TWT Controller Python Binding Interface")

# Python Bindings Library Sources with pb-twt- prefix
set(TWT_PYTHON_BINDINGS_SOURCES
    pb-twt-wrapper-ns3.cc
)

set(TWT_PYTHON_BINDINGS_HEADERS
    pb-twt-wrapper-ns3.h
    pb-twt-core.h
)

# Print author information
message(STATUS "========================================")
message(STATUS "  TWT Controller Python Bindings Module")
message(STATUS "  Author: ${PROJECT_AUTHOR}")
message(STATUS "  Email: ${PROJECT_EMAIL}")
message(STATUS "  Lab: ${PROJECT_LAB}")
message(STATUS "  PI: ${PROJECT_PI}")
message(STATUS "  Description: ${PROJECT_DESCRIPTION}")
message(STATUS "========================================")

# Build TWT main simulation with modular components
build_lib_example(
  NAME twt-main-simulation
  SOURCE_FILES 
    twt-main-simulation.cc
    twt-simulation-config.cc
    twt-trace-callbacks.cc
    twt-metrics.cc
    pb-twt-wrapper-ns3.cc
  LIBRARIES_TO_LINK
    ${libai}
    ${libcore}
    ${libnetwork}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libspectrum}
    ${libpropagation}
    ${libantenna}
    ${libenergy}
    ${libflow-monitor}
    ${libpoint-to-point}
    ${libinternet-apps}
)

# Python binding module using pybind11 with pb-twt- naming
pybind11_add_module(pb_twt_interface_py pb-twt-interface.cc)
set_target_properties(pb_twt_interface_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_NAME "pb_twt_interface_py"
    PREFIX ""
    SUFFIX ".so"
)
target_link_libraries(pb_twt_interface_py PRIVATE ${libai})

# Build Python interface along with C++ lib
# Note: Dependencies will be handled automatically by the build system

# Additional deployment for Python binding library - copy to cmake-cache and source
add_custom_command(TARGET pb_twt_interface_py POST_BUILD
    # Copy to cmake-cache build directory
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:pb_twt_interface_py>
    ${CMAKE_BINARY_DIR}/contrib/ai/examples/twt/pb_twt_interface_py.so
    
    # Keep in source directory as well
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:pb_twt_interface_py>
    ${CMAKE_CURRENT_SOURCE_DIR}/pb_twt_interface_py.so
    
    COMMENT "Copying TWT Python binding library by ${PROJECT_AUTHOR}"
)

# Create documentation target
add_custom_target(pb_twt_docs
    COMMAND echo "========================================="
    COMMAND echo "TWT Controller Python Bindings Documentation"
    COMMAND echo "Author: ${PROJECT_AUTHOR}"
    COMMAND echo "Email: ${PROJECT_EMAIL}"
    COMMAND echo "Lab: ${PROJECT_LAB}"
    COMMAND echo "PI: ${PROJECT_PI}"
    COMMAND echo "========================================="
    COMMAND echo "Files created:"
    COMMAND echo "  - pb-twt-core.h: TWT data structures (EnvStruct, ActionStruct)"
    COMMAND echo "  - pb-twt-wrapper-ns3.h/cc: TWT wrapper class (TWTWrapper) for C++ integration"
    COMMAND echo "  - pb-twt-interface.cc: Python bindings module (pb_twt_interface_py)"
    COMMAND echo "  - pb_twt_wrapper_py.py: Python NS3-AI wrapper (TWTWrapper) for reset/step/close"
    COMMAND echo ""
    COMMAND echo "Executables built:"
    COMMAND echo "  - twt-main-simulation: TWT RL simulation executable"
    COMMAND echo "========================================="
    COMMENT "Generating TWT documentation by ${PROJECT_AUTHOR}"
)

# Export variables for use by parent CMakeLists
set(TWT_PYTHON_BINDINGS_SOURCES ${TWT_PYTHON_BINDINGS_SOURCES} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_HEADERS ${TWT_PYTHON_BINDINGS_HEADERS} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_AUTHOR "${PROJECT_AUTHOR}" PARENT_SCOPE)

# Add compile definitions for author information
# Note: Compile definitions will be handled by the build system automatically

# Installation rules
install(FILES ${TWT_PYTHON_BINDINGS_HEADERS}
    DESTINATION include/ns3/ai/twt-bindings
    COMPONENT development
)

install(TARGETS pb_twt_interface_py
    LIBRARY DESTINATION lib/twt-bindings
    COMPONENT runtime
)
