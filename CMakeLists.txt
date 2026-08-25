cmake_minimum_required(VERSION 3.10)
project(rk1828_qwen35_9b_2cards_service)

# The service intentionally follows the Rockchip multicard demo.  Point this
# at the SDK checkout instead of copying SDK sources into this repository.
set(RKNN3_MODEL_ZOO_ROOT "" CACHE PATH "Path to rknn3-model-zoo")
if(NOT RKNN3_MODEL_ZOO_ROOT)
  message(FATAL_ERROR "Set -DRKNN3_MODEL_ZOO_ROOT=/path/to/rknn3-model-zoo")
endif()

add_subdirectory(${RKNN3_MODEL_ZOO_ROOT}/3rdparty thirdparty.out)

add_executable(rk1828_qwen35_9b_2cards_daemon
  native/src/main.cc
)

target_compile_features(rk1828_qwen35_9b_2cards_daemon PRIVATE cxx_std_14)
target_include_directories(rk1828_qwen35_9b_2cards_daemon PRIVATE
  ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/rknpu3/include
  ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/tokenizer/include
  ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/json
)
target_link_libraries(rk1828_qwen35_9b_2cards_daemon PRIVATE
  ${LIBRKNNAPI}
  ${LIBTOKENIZER}
  dl
  pthread
)
set_target_properties(rk1828_qwen35_9b_2cards_daemon PROPERTIES
  INSTALL_RPATH "$ORIGIN/lib"
)
install(TARGETS rk1828_qwen35_9b_2cards_daemon DESTINATION bin)
install(FILES ${LIBRKNNAPI} DESTINATION lib)
