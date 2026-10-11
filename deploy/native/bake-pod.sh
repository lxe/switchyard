#!/bin/bash
set -euo pipefail
printf '{"stage":"container_started","purpose":"native_offline_bake","wall_s":%s}\n' "$(date +%s.%N)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install --no-install-recommends -y openssh-server zstd
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "$SWITCHYARD_SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
printf '{"stage":"bake_control_listener","wall_s":%s}\n' "$(date +%s.%N)"
exec /usr/sbin/sshd -D -e
