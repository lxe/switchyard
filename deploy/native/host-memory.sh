#!/bin/bash
# Fixed read-only telemetry from this process's memory cgroup. No Python.
native_memory_number() {
  local value
  if [ -r "$1" ] && read -r value < "$1" && [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s' "$value"
  else
    printf 'null'
  fi
}

native_host_memory_sample() {
  local hierarchy controllers group base='' version='' scope='process_cgroup'
  local current peak limit swap events oom='null' kills='null' name value
  while IFS=: read -r hierarchy controllers group; do
    if [ "$hierarchy" = 0 ] && [ -z "$controllers" ]; then
      base="/sys/fs/cgroup${group%/}"
      version='v2'
      break
    fi
    if [[ ",$controllers," = *,memory,* ]]; then
      base="/sys/fs/cgroup/memory${group%/}"
      version='v1'
    fi
  done < /proc/self/cgroup
  if [ "$version" = v2 ]; then
    if [ ! -r "$base/memory.current" ]; then
      base='/sys/fs/cgroup'; scope='mount_root_fallback'
    fi
    current="$base/memory.current"; peak="$base/memory.peak"
    limit="$base/memory.max"; swap="$base/memory.swap.current"
    events="$base/memory.events"
  elif [ "$version" = v1 ]; then
    if [ ! -r "$base/memory.usage_in_bytes" ]; then
      base='/sys/fs/cgroup/memory'; scope='mount_root_fallback'
    fi
    current="$base/memory.usage_in_bytes"; peak="$base/memory.max_usage_in_bytes"
    limit="$base/memory.limit_in_bytes"; swap='/nonexistent'
    events="$base/memory.oom_control"
  else
    version='unavailable'; scope='unavailable'
    current='/nonexistent'; peak='/nonexistent'; limit='/nonexistent'
    swap='/nonexistent'; events='/nonexistent'
  fi
  if [ -r "$events" ]; then
    while read -r name value; do
      [[ "$value" =~ ^[0-9]+$ ]] || continue
      case "$name" in
        oom) oom="$value";;
        oom_kill) kills="$value";;
      esac
    done < "$events"
  fi
  printf 'native_host_memory={"wall_s":%s,"cgroup_version":"%s","scope":"%s","current_bytes":%s,"peak_bytes":%s,"limit_bytes":%s,"swap_current_bytes":%s,"oom":%s,"oom_kill":%s}\n' \
    "$(date +%s.%N)" "$version" "$scope" \
    "$(native_memory_number "$current")" "$(native_memory_number "$peak")" \
    "$(native_memory_number "$limit")" "$(native_memory_number "$swap")" "$oom" "$kills"
}

native_host_memory_monitor() {
  local sample
  for ((sample=0; sample<600; sample++)); do
    native_host_memory_sample
    sleep 1
  done
}
