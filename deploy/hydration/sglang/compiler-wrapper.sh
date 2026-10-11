#!/bin/sh
# Relocated Ubuntu toolchain, confined to the serving artifact.
set -eu
root=/opt/switchyard/runtime/toolchain
name=$(basename "$0")
export GCC_EXEC_PREFIX="$root/lib/gcc/x86_64-linux-gnu/13/"
export CPLUS_INCLUDE_PATH="$root/sysroot/usr/include/c++/13:$root/sysroot/usr/include/x86_64-linux-gnu/c++/13${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
exec "$root/bin/$name.real" --sysroot="$root/sysroot" \
  -B"$root/lib/gcc/x86_64-linux-gnu/13/" -B"$root/bin/" "$@"
