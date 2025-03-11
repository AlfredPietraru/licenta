cmake_minimum_required( VERSION 3.8 )
project( main )
# set( CMAKE_CXX_FLAGS "-std=c++11" )
set(CMAKE_CXX_FLAGS  "-std=c++17")
set(OpenGL_GL_PREFERENCE LEGACY)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(fmt REQUIRED)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)
include_directories ( ${EIGEN3_INCLUDE_DIR} )

find_package(Ceres REQUIRED)
include_directories(${CERES_LIBRARIES})

set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

# find_package(G2O REQUIRED)
# set(G2O_LIBS
#     g2o_core
#     g2o_stuff
#     g2o_solver_csparse
#     g2o_types_slam3d
#     g2o_types_slam2d
# )
# include_directories(${G2O_INCLUDE_DIRS})


find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
FIND_PACKAGE(DBoW3 REQUIRED) 
include_directories(${DBoW3_LIBS})


set(SOURCES
    main.cpp
    src/BundleAdjustment.cpp
    src/FeatureFinderMatcher.cpp
    src/Map.cpp
    src/Tracker.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
)

set(HEADERS
    "include/BundleAdjustment.h"
    "include/FeatureFinderMatcher.h"
    "include/Map.h"
    "include/Tracker.h"
    "include/MapPoint.h"
    "include/KeyFrame.h"
)

add_executable(main ${SOURCES} ${HEADERS})

target_link_libraries(main
    fmt::fmt
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Sophus_LIBRARIES}
    # ${DBoW3_LIBS}
)

# /usr/local/lib/libg2o_core.so
# /usr/local/lib/libg2o_stuff.so
# /usr/local/lib/libg2o_solver_csparse.so
# /usr/local/lib/libg2o_types_slam3d.so
# /usr/local/lib/libg2o_types_slam2d.so
