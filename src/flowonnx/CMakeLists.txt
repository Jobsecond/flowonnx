if(FLOWONNX_BUILD_STATIC)
    add_library(${PROJECT_NAME} STATIC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLOWONNX_STATIC)
else()
    add_library(${PROJECT_NAME} SHARED)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${PROJECT_NAME} PRIVATE FLOWONNX_LIBRARY)

if (ONNXRUNTIME_ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_ENABLE_CUDA)
endif()

if (ONNXRUNTIME_ENABLE_DML)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_ENABLE_DML)
endif()

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

file(GLOB_RECURSE _src *.h *.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${_src})

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

target_include_directories(${PROJECT_NAME} PRIVATE . private)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# OnnxRuntime
target_include_directories(${PROJECT_NAME} PRIVATE ../../libs/onnxruntime/include)
target_link_directories(${PROJECT_NAME} PRIVATE ../../libs/onnxruntime/lib)
target_compile_definitions(${PROJECT_NAME} PRIVATE ORT_API_MANUAL_INIT)

if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME}
        COMMAND xcopy /Q /Y *.dll \"$<TARGET_FILE_DIR:${PROJECT_NAME}>/../bin\" > NUL
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/onnxruntime/lib"
    )
elseif(APPLE)
    add_custom_command(TARGET ${PROJECT_NAME}
        COMMAND cp -f *.dylib \"$<TARGET_FILE_DIR:${PROJECT_NAME}>/../lib\"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/onnxruntime/lib"
    )
else()
    add_custom_command(TARGET ${PROJECT_NAME}
        COMMAND cp -f *.so \"$<TARGET_FILE_DIR:${PROJECT_NAME}>/../lib\"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/onnxruntime/lib"
    )
endif()

# Libraries
find_package(loadso CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE loadso::loadso)

# Add platform specific
if(WIN32)
    set(RC_DESCRIPTION "OnnxRuntime Inference Flow Utility")
    set(RC_COPYRIGHT "Copyright (C) 2024 Team OpenVPI")
    include("../../cmake/winrc.cmake")
endif()

if(FLOWONNX_INSTALL)
    target_include_directories(${PROJECT_NAME} PUBLIC
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    install(TARGETS ${PROJECT_NAME}
        EXPORT ${FLOWONNX_INSTALL_NAME}Targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" OPTIONAL
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" OPTIONAL
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" OPTIONAL
    )

    # Install headers
    file(GLOB _headers *.h)
    install(FILES ${_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${FLOWONNX_INSTALL_NAME})
endif()

if(FLOWONNX_INSTALL)
    # Add install target
    set(_install_dir ${CMAKE_INSTALL_LIBDIR}/cmake/${FLOWONNX_INSTALL_NAME})

    # Add version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${FLOWONNX_INSTALL_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # Add configuration file
    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/${FLOWONNX_INSTALL_NAME}Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${FLOWONNX_INSTALL_NAME}Config.cmake"
        INSTALL_DESTINATION ${_install_dir}
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    # Install cmake files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${FLOWONNX_INSTALL_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${FLOWONNX_INSTALL_NAME}ConfigVersion.cmake"
        DESTINATION ${_install_dir}
    )

    # Install cmake targets files
    install(EXPORT ${FLOWONNX_INSTALL_NAME}Targets
        FILE "${FLOWONNX_INSTALL_NAME}Targets.cmake"
        NAMESPACE ${FLOWONNX_INSTALL_NAME}::
        DESTINATION ${_install_dir}
    )
endif()