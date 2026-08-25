cmake_minimum_required(VERSION 3.10)
project(rk1828_qwen35_9b_2cards_service)

# The service intentionally follows the Rockchip multicard demo.  Point this
# at the SDK checkout instead of copying SDK sources into this repository.
set(RKNN3_MODEL_ZOO_ROOT "" CACHE PATH "Path to rknn3-model-zoo")
if(NOT RKNN3_MODEL_ZOO_ROOT)
  message(FATAL_ERROR "Set -DRKNN3_MODEL_ZOO_ROOT=/path/to/rknn3-model-zoo")
endif()
set(TARGET_SOC "rk3588" CACHE STRING "Host SoC used by the RK1828 PCIe runtime")

add_subdirectory(${RKNN3_MODEL_ZOO_ROOT}/3rdparty thirdparty.out)

# Allow a newer RKNN3 API / tokenizer package to override the copies bundled in
# an older model-zoo checkout, matching the upstream multicard example.
set(RKNN3_API_ROOT "" CACHE PATH "Directory containing include/ and Linux/<arch>/ for the RKNN3 API")
if(RKNN3_API_ROOT)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_RKNN3_API_ARCH aarch64)
  else()
    set(_RKNN3_API_ARCH ${TARGET_LIB_ARCH})
  endif()
  set(LIBRKNNAPI_INCLUDES ${RKNN3_API_ROOT}/include)
  set(LIBRKNNAPI ${RKNN3_API_ROOT}/Linux/${_RKNN3_API_ARCH}/librknn3_api.so)
endif()

set(TOKENIZER_ROOT "" CACHE PATH "Directory containing include/ and lib/Linux/<arch>/ for tokenizer")
if(TOKENIZER_ROOT)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_TOKENIZER_ARCH aarch64)
  else()
    set(_TOKENIZER_ARCH ${TARGET_LIB_ARCH})
  endif()
  set(LIBTOKENIZER_INCLUDES ${TOKENIZER_ROOT}/include)
  set(LIBTOKENIZER ${TOKENIZER_ROOT}/lib/Linux/${_TOKENIZER_ARCH}/libtokenizer.a)
endif()

if(NOT LIBRKNNAPI_INCLUDES)
  set(LIBRKNNAPI_INCLUDES ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/rknpu3/include)
endif()
if(NOT LIBTOKENIZER_INCLUDES)
  set(LIBTOKENIZER_INCLUDES ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/tokenizer/include)
endif()

add_executable(rk1828_qwen35_9b_2cards_daemon
  native/src/main.cc
)

target_compile_features(rk1828_qwen35_9b_2cards_daemon PRIVATE cxx_std_14)
target_include_directories(rk1828_qwen35_9b_2cards_daemon PRIVATE
  ${LIBRKNNAPI_INCLUDES}
  ${LIBTOKENIZER_INCLUDES}
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
install(PROGRAMS ${LIBRKNNAPI} DESTINATION lib)
