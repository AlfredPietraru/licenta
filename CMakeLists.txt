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

find_package(Pangolin REQUIRED)


set( DBoW2_INCLUDE_DIRS "/usr/local/include/DBoW2" )
set( DBoW2_LIBS "/usr/local/lib/libDBoW2.so" )

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

set(G2O_LIBS
    g2o_core
    g2o_stuff
    g2o_solver_csparse
    g2o_solver_dense
    g2o_types_slam3d
    g2o_types_slam2d
    g2o_types_sim3
    g2o_types_sba
)


find_package(Ceres REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
FIND_PACKAGE(DBoW2 REQUIRED) 
FIND_PACKAGE(yaml-cpp REQUIRED)
include_directories(${DBoW2_INCLUDE_DIRS})


set(SOURCES
    main.cpp
    src/MotionOnlyBA.cpp
    src/Feature.cpp
    src/FeatureFinderMatcher.cpp
    src/Map.cpp
    src/Tracker.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
    src/OrbMatcher.cpp
    src/config.cpp
    src/TumDatasetReader.cpp
    src/LocalMapping.cpp
    src/ORBextractor.cpp
    src/MapDrawer.cpp
)

set(HEADERS
    "include/MotionOnlyBA.h"
    "include/Feature.h"
    "include/FeatureFinderMatcher.h"
    "include/Map.h"
    "include/Tracker.h"
    "include/MapPoint.h"
    "include/KeyFrame.h"
    "include/OrbMatcher.h"
    "include/config.h"
    "include/TumDatasetReader.h"
    "include/ORBVocabulary.h"
    "include/LocalMapping.h"
    "include/ORBextractor.h"
    "include/MapDrawer.h"
)


add_executable(main ${SOURCES} ${HEADERS})

target_link_libraries(main
    fmt::fmt
    yaml-cpp
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${DBoW2_LIBS}
    ${Pangolin_LIBRARIES}
    ${G2O_LIBS}
)

# /usr/local/lib/libg2o_core.so
# /usr/local/lib/libg2o_stuff.so
# /usr/local/lib/libg2o_solver_csparse.so
# /usr/local/lib/libg2o_types_slam3d.so
# /usr/local/lib/libg2o_types_slam2d.so
