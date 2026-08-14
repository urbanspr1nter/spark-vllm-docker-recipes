#!/bin/bash

if [[ -z "${1:-}" ]]; then
  echo "Error: IP suffix is required." >&2
  echo "Usage: $0 <ip-suffix>" >&2
  exit 1
fi

IP_SUFFIX=$1

sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.100.$IP_SUFFIX/24
      dhcp4: no
      mtu: 9000
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      optional: true
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply