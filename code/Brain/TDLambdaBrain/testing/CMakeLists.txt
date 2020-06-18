cmake_minimum_required ( VERSION 3.13.3 )
project ( brain_test )
set(EXE test)
add_executable( ${EXE} test.cpp )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_target_properties(${EXE} PROPERTIES
            CXX_STANDARD 20
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON
            )

target_include_directories(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Global.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../module_factories.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Genome/CircularGenome/CircularGenome.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Utilities/Data.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Utilities/Parameters.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Utilities/MTree.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Utilities/CSV.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Archivist/DefaultArchivist.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Brain/AbstractBrain.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Optimizer/AbstractOptimizer.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Optimizer/RouletteOptimizer/RouletteOptimizer.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Genome/AbstractGenome.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Group/Group.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Organism/Organism.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../World/AbstractWorld.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../World/ConveyorBeltWorld/ConveyorBeltWorld.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Brain/TDLambdaBrain/TDLambdaBrain.cpp)
target_sources(${EXE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../Brain/TDLambdaBrain/TDLambda.cpp)
find_package( Threads )
target_link_libraries( test ${CMAKE_THREAD_LIBS_INIT} )
