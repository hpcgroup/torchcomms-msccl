#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -x

# Perlmutter: HOME may be unset in some environments (e.g. Claude Code).
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(whoami)" | cut -d: -f6)"
  export HOME
fi

function do_cmake_build() {
  local source_dir="$1"
  local extra_flags="$2"
  # Clear stale cmake cache so the shared build-output dir can be
  # reconfigured for a different source tree.
  rm -f CMakeCache.txt
  rm -rf CMakeFiles
  cmake -G Ninja \
    -DCMAKE_C_COMPILER="${CC_FOR_BUILD:-$(command -v gcc)}" \
    -DCMAKE_CXX_COMPILER="${CXX_FOR_BUILD:-$(command -v g++)}" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_MODULE_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_DIR="$INSTALL_PREFIX" \
    -DBIN_INSTALL_DIR="$INSTALL_PREFIX/bin" \
    -DLIB_INSTALL_DIR="$INSTALL_PREFIX/$LIB_SUFFIX" \
    -DINCLUDE_INSTALL_DIR="$INSTALL_PREFIX/include" \
    -DCMAKE_INSTALL_INCLUDEDIR="$INSTALL_PREFIX/include" \
    -DCMAKE_INSTALL_LIBDIR="$INSTALL_PREFIX/$LIB_SUFFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.22 \
    $extra_flags \
    -S "${source_dir}"
  ninja -j4
  ninja -j4 install
}

function clean_third_party {
  local library_name="$1"
  if [[ -z "${CONDA_PREFIX}" ]]; then
    echo "CONDA_PREFIX is empty; skip clean_third_party for ${library_name}"
    return
  fi
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -rf "${CONDA_PREFIX}"/include/"${library_name}"*/
    rm -rf "${CONDA_PREFIX}"/include/"${library_name}"*.h
  fi
}

function build_fb_oss_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  local source_dir="../${library_name}/${library_name}"
  if [ -f ${library_name}/CMakeLists.txt ]; then
    source_dir="../${library_name}"
  fi
  if [ -f ${library_name}/build/cmake/CMakeLists.txt ]; then
    source_dir="../${library_name}/build/cmake"
  fi
  if [ -f ${library_name}/cmake_unofficial/CMakeLists.txt ]; then
    source_dir="../${library_name}/cmake_unofficial"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  if [[ "${CLEAN_THIRD_PARTY}" == 1 || "${INCREMENTAL_BUILD}" == 0 ]]; then
    rm -rf build-output
  fi
  mkdir -p build-output
  pushd build-output
  do_cmake_build "$source_dir" "$extra_flags"
  popd
}

function build_automake_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
  ./configure --prefix="$INSTALL_PREFIX" --disable-pie

  make -j
  make install
  popd
}

function build_boost() {
  local repo_url="https://github.com/boostorg/boost.git"
  local repo_tag="boost-1.82.0"
  local library_name="boost"
  local extra_flags=""

  # clean up existing boost
  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
  ./bootstrap.sh --prefix="$INSTALL_PREFIX" --libdir="$INSTALL_PREFIX/$LIB_SUFFIX" --without-libraries=python
  ./b2 -q cxxflags=-fPIC cflags=-fPIC install
  popd
}

function build_openssl() {
  local repo_url="https://github.com/openssl/openssl.git"
  local repo_tag="openssl-3.5.1"
  local library_name="openssl"
  local extra_flags=""

  # clean up existing boost
  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  pushd "$library_name"
  ./config no-shared --prefix="$INSTALL_PREFIX" --openssldir="$INSTALL_PREFIX" --libdir=lib

  make -j
  make install
  popd
}

function build_third_party {
  # build third-party libraries
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -f "${CONDA_PREFIX}"/*.cmake 2>/dev/null || true
  fi
  local third_party_tag="v2026.01.19.00"

  mkdir -p /tmp/third-party
  pushd /tmp/third-party
  if [[ -z "${USE_SYSTEM_LIBS}" ]]; then
    build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF"
    build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON"
    build_fb_oss_library "https://github.com/madler/zlib.git" "v1.2.13" zlib "-DZLIB_BUILD_TESTING=OFF"
    build_boost
    build_openssl
    build_fb_oss_library "https://github.com/Cyan4973/xxHash.git" "v0.8.0" xxhash
    # we need both static and dynamic gflags since thrift generator can't
    # statically link against glog.
    build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags
    build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags "-DBUILD_SHARED_LIBS=ON"
    # we need both static and dynamic glog since thrift generator can't
    # statically link against glog.
    build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog
    build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog "-DBUILD_SHARED_LIBS=ON"
    build_fb_oss_library "https://github.com/facebook/zstd.git" "v1.5.6" zstd
    build_automake_library "https://github.com/jedisct1/libsodium.git" "1.0.20-RELEASE" sodium
    build_fb_oss_library "https://github.com/fastfloat/fast_float.git" "v8.0.2" fast_float "-DFASTFLOAT_INSTALL=ON"
    build_fb_oss_library "https://github.com/libevent/libevent.git" "release-2.1.12-stable" event "-DEVENT__DISABLE_TESTS=ON -DEVENT__DISABLE_BENCHMARK=ON -DEVENT__DISABLE_SAMPLES=ON"
    build_fb_oss_library "https://github.com/google/double-conversion.git" "v3.3.1" double-conversion
    build_fb_oss_library "https://github.com/facebook/folly.git" "$third_party_tag" folly "-DUSE_STATIC_DEPS_ON_UNIX=ON -DOPENSSL_USE_STATIC_LIBS=ON"
  else
    if [[ "${NCCL_USE_CONDA}" == 1 && -z "${NCCL_SKIP_CONDA_INSTALL}" ]]; then
      DEPS=(
        cmake
        ninja
        jemalloc
        gtest
        boost
        double-conversion
        libevent
        conda-forge::libsodium
        libunwind
        snappy
        conda-forge::fast_float
        libdwarf-dev
        gflags
        glog==0.4.0
        xxhash
        zstd
        conda-forge::zlib
        conda-forge::libopenssl-static
        conda-forge::folly
        fmt
      )
      conda install "${DEPS[@]}" --yes
    else
      echo "USE_SYSTEM_LIBS=1 and NCCL_USE_CONDA!=1; skipping conda install"
    fi
  fi

  # TODO: migrate out all dependencies for feedstock
  if [[ -z "${NCCL_FEEDSTOCK_BUILD}" ]]; then
    build_fb_oss_library "https://github.com/facebookincubator/fizz.git" "$third_party_tag" fizz "-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF"
    build_fb_oss_library "https://github.com/facebook/mvfst" "$third_party_tag" quic
    build_fb_oss_library "https://github.com/facebook/wangle.git" "$third_party_tag" wangle "-DBUILD_TESTS=OFF"
  fi
  build_fb_oss_library "https://github.com/facebook/fbthrift.git" "$third_party_tag" thrift
  popd
}

function build_comms_tracing_service {
  local include_prefix="comms/analyzer/if"
  local base_dir="${PWD}"
  local build_dir=/tmp/build/comms_tracing_service

  mkdir -p "$build_dir"
  pushd "$build_dir"
  # set up the directory structure
  mkdir -p "$include_prefix"
  cp -r "${base_dir}/${include_prefix}"/* "$include_prefix"
  mv "$include_prefix"/CMakeLists.txt .

  # set up the build config
  cp -r /tmp/third-party/thrift/build .
  # Generate a minimal FBThriftConfig.cmake that points at the install prefix.
  # The installed version uses PACKAGE_PREFIX_DIR relative to its own location,
  # which breaks when copied elsewhere.
  cat > build/fbcode_builder/CMake/FBThriftConfig.cmake <<FBCFG
set(FBTHRIFT_INCLUDE_DIR "${INSTALL_PREFIX}/include")
set(FBTHRIFT_COMPILER "${INSTALL_PREFIX}/bin/thrift1")
find_package(Xxhash REQUIRED)
find_package(ZLIB REQUIRED)
find_package(mvfst CONFIG REQUIRED)
if (NOT TARGET FBThrift::thriftcpp2)
  include("${INSTALL_PREFIX}/FBThriftTargets.cmake")
endif()
set(FBThrift_FOUND True)
FBCFG

  # build the thrift service library
  cd build
  do_cmake_build ..

  popd
}

if [ -z "$DEV_SIGNATURE" ]; then
    is_git=$(git rev-parse --is-inside-work-tree)
    if [ $is_git ]; then
        DEV_SIGNATURE="git-"$(git rev-parse --short HEAD)
    else
        echo "Cannot detect source repository hash. Skip"
        DEV_SIGNATURE=""
    fi
fi

set -e
export LIB_PREFIX="lib64"

BUILDDIR=${BUILDDIR:="${PWD}/build/ncclx"}
CUDA_HOME=${CUDA_HOME:="/usr/local/cuda"}
# Perlmutter: auto-detect CUDA_HOME from nvcc if the default doesn't exist
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  _nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$_nvcc_path" ]]; then
    CUDA_HOME="$(dirname "$(dirname "$_nvcc_path")")"
  fi
  unset _nvcc_path
fi
NVCC_ARCH=${NVCC_ARCH:="a100"}

# Add b200 support if CUDA 12.8+ is available
CUDA_VERSION=$("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
if [[ -n "${NVCC_ARCH_ADD_B200:-}" ]] && ([[ "$CUDA_MAJOR" -gt 12 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]); then
    NVCC_ARCH="${NVCC_ARCH},b200"
fi
NCCL_FP8=${NCCL_FP8:=1}
NCCL_DEBUG_BUILD=${NCCL_DEBUG_BUILD:=0}
CLEAN_BUILD=${CLEAN_BUILD:=0}
INCREMENTAL_BUILD=${INCREMENTAL_BUILD:=1}

# Choose a writable install prefix.  Use an explicit INSTALL_PREFIX or
# CONDA_PREFIX if provided; otherwise fall back to a user-writable directory.
INSTALL_PREFIX="${INSTALL_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -z "${INSTALL_PREFIX}" ]]; then
  if [[ -n "${PSCRATCH:-}" ]]; then
    INSTALL_PREFIX="${PSCRATCH}/ncclx-deps"
  elif [[ -n "${SCRATCH:-}" ]]; then
    INSTALL_PREFIX="${SCRATCH}/ncclx-deps"
  else
    INSTALL_PREFIX="${HOME}/.local/ncclx-deps"
  fi
fi
export CONDA_PREFIX="${INSTALL_PREFIX}"
if [[ -z "${INSTALL_PREFIX}" || "${INSTALL_PREFIX}" == "/" ]]; then
  echo "ERROR: INSTALL_PREFIX is empty or '/'. Set INSTALL_PREFIX or CONDA_PREFIX to a writable path."
  exit 1
fi
mkdir -p "${INSTALL_PREFIX}"

# Load cmake module if available (must happen AFTER INSTALL_PREFIX is set so
# module-injected CMAKE_PREFIX_PATH does not leak into INSTALL_PREFIX).
if command -v module &>/dev/null; then
  module load cmake 2>/dev/null || true
fi

# Build CMAKE_PREFIX_PATH: our install prefix first, then anything the
# environment (or modules) already added.
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH}"
else
  export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
fi

if [[ -z "${LIB_SUFFIX:-}" ]]; then
  if [[ -d "${INSTALL_PREFIX}/lib64" ]]; then
    LIB_SUFFIX=lib64
  else
    LIB_SUFFIX=lib
  fi
fi
CONDA_INCLUDE_DIR="${INSTALL_PREFIX}/include"
CONDA_LIB_DIR="${INSTALL_PREFIX}/${LIB_SUFFIX}"
NCCL_HOOK_LIBS=${NCCL_HOOK_LIBS:=0}
NCCL_HOME=${NCCL_HOME:="${PWD}/comms/ncclx/stable"}
BASE_DIR=${BASE_DIR:="${PWD}"}
CUDARTLIB=cudart_static
THIRD_PARTY_LDFLAGS=""

if [[ -z "${NCCL_BUILD_SKIP_DEPS}" ]]; then
  echo "Building dependencies"
  build_third_party
fi
if [[ -z "${NCCL_BUILD_SKIP_DEPS}" || -n "${NCCL_BUILD_TRACING_SERVICE}" ]]; then
  build_comms_tracing_service
fi

# Generate nccl_cvars files (these are no longer checked into the repo)
# The files are generated by extractcvars.py which reads nccl_cvars.yaml and nccl_cvars.cc.in
echo "Generating nccl_cvars files..."
CVARS_DIR="$BASE_DIR/comms/utils/cvars"

# Validate that the required source files exist
if [ ! -f "$CVARS_DIR/extractcvars.py" ]; then
  echo "ERROR: extractcvars.py not found at $CVARS_DIR/extractcvars.py"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.yaml" ]; then
  echo "ERROR: nccl_cvars.yaml not found at $CVARS_DIR/nccl_cvars.yaml"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.cc.in" ]; then
  echo "ERROR: nccl_cvars.cc.in not found at $CVARS_DIR/nccl_cvars.cc.in"
  exit 1
fi

# Ensure PyYAML is available for extractcvars.py (no conda requirement).
python3 - <<'PY'
try:
  import yaml  # noqa: F401
except Exception as e:
  raise SystemExit("PyYAML not found. Install it in your Python env (e.g., `python3 -m pip install pyyaml`).")
PY

# Run the extractcvars.py script directly to generate the files
export NCCL_CVARS_OUTPUT_DIR="$CVARS_DIR"
python3 "$CVARS_DIR/extractcvars.py"

# Verify the files were generated
if [ ! -f "$CVARS_DIR/nccl_cvars.h" ] || [ ! -f "$CVARS_DIR/nccl_cvars.cc" ]; then
  echo "ERROR: Failed to generate nccl_cvars files"
  exit 1
fi
echo "Successfully generated nccl_cvars files in $CVARS_DIR"

# set up the third-party ldflags
export PKG_CONFIG_PATH="${CONDA_LIB_DIR}"/pkgconfig
THRIFT_SERVICE_LDFLAGS=(
  "-l:libcomms_tracing_service.a"
  "-Wl,--start-group"
  "-l:libasync.a"
  "-l:libconcurrency.a"
  "-l:libthrift-core.a"
  "-l:libthriftanyrep.a"
  "-l:libthriftcpp2.a"
  "-l:libthriftmetadata.a"
  "-l:libthriftprotocol.a"
  "-l:libthrifttype.a"
  "-l:libthrifttyperep.a"
  "-l:librpcmetadata.a"
  "-l:libruntime.a"
  "-l:libserverdbginfo.a"
  "-l:libtransport.a"
  "-l:libcommon.a"
  "-Wl,--end-group"
  "-l:libwangle.a"
  "-l:libfizz.a"
  "-l:libxxhash.a"
)
THIRD_PARTY_LDFLAGS+="${THRIFT_SERVICE_LDFLAGS[*]} "
THIRD_PARTY_LDFLAGS+="$(pkg-config --libs --static libfolly) "

if [[ -z "${USE_SYSTEM_LIBS}" ]]; then
  THIRD_PARTY_LDFLAGS+="-l:libglog.a -l:libgflags.a -l:libboost_context.a -l:libfmt.a -l:libssl.a -l:libcrypto.a "
else
  THIRD_PARTY_LDFLAGS+="-lglog -lgflags -lboost_context -lfmt -lssl -lcrypto "
fi

# ---------------------------------------------------------------------------
# ADDED FIX FOR RTLD_DEEPBIND / PyTorch Symbol Collisions
# Forces internal library symbols to bind locally during the link step.
# ---------------------------------------------------------------------------
THIRD_PARTY_LDFLAGS+="-Wl,-Bsymbolic "

echo "$THIRD_PARTY_LDFLAGS"

if [[ -z "${NVCC_GENCODE-}" ]]; then
    IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
    arch_gencode=""
    for arch in "${arch_array[@]}"
    do
        case "$arch" in
        "p100")
        arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
            ;;
        "v100")
        arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
            ;;
        "a100")
        arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
        ;;
        "h100")
            arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
        ;;
        "b200")
            arch_gencode="$arch_gencode -gencode=arch=compute_100,code=sm_100"
        ;;
        esac
    done
    NVCC_GENCODE=$arch_gencode
fi

if [ "$CLEAN_BUILD" == 1 ]; then
    rm -rf "$BUILDDIR"
fi

mkdir -p "$BUILDDIR"
pushd "${NCCL_HOME}"

function build_nccl {
  local debug_flag=""
  if [[ "$NCCL_DEBUG_BUILD" == "1" ]]; then
    debug_flag="DEBUG=1"
    echo "=== DEBUG BUILD (-O0 -G -g) ==="
  fi
  make VERBOSE=1 -j4 \
    src.build \
    BUILDDIR="$BUILDDIR" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    NCCL_SUFFIX="x-${DEV_SIGNATURE}" \
    NCCL_FP8="$NCCL_FP8" \
    BASE_DIR="$BASE_DIR" \
    CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
    CONDA_LIB_DIR="$CONDA_LIB_DIR" \
    THIRD_PARTY_LDFLAGS="$THIRD_PARTY_LDFLAGS" \
    CUDARTLIB="$CUDARTLIB" \
    $debug_flag
}

function build_and_install_nccl {
  local debug_flag=""
  if [[ "$NCCL_DEBUG_BUILD" == "1" ]]; then
    debug_flag="DEBUG=1"
    echo "=== DEBUG BUILD (-O0 -G -g) ==="
  fi
  make VERBOSE=1 -j4 \
    src.install \
    BUILDDIR="$BUILDDIR" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    NCCL_SUFFIX="x-${DEV_SIGNATURE}" \
    NCCL_FP8="$NCCL_FP8" \
    BASE_DIR="$BASE_DIR" \
    CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
    CONDA_LIB_DIR="$CONDA_LIB_DIR" \
    THIRD_PARTY_LDFLAGS="$THIRD_PARTY_LDFLAGS" \
    CUDARTLIB="$CUDARTLIB" \
    $debug_flag
}

if [[ -z "${NCCL_BUILD_AND_INSTALL}" ]]; then
  build_nccl
else
  build_and_install_nccl
fi

# sanity check
if [ -n "${NCCL_RUN_SANITY_CHECK}" ]; then
    pushd examples
    export NCCL_DEBUG=WARN
    export LD_LIBRARY_PATH=$BUILDDIR/lib

    make all \
      NVCC_GENCODE="$NVCC_GENCODE" \
      CUDA_HOME="$CUDA_HOME" \
      NCCL_HOME="$CONDA_PREFIX"

    set +e

    TIMEOUT=10s
    timeout $TIMEOUT "$BUILDDIR"/examples/HelloWorld
    if [ "$?" == "124" ]; then
        echo "Program TIMEOUT in ${TIMEOUT}. Terminate."
    fi
    popd
fi

popd
