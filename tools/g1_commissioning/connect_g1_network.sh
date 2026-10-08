#!/usr/bin/env bash
set -euo pipefail

nic="${1:-enx6c1ff701509c}"
connection_uuid="14bd2aef-6a15-4380-9752-cd03f3fa528b"
expected_address="192.168.123.99/24"

if [[ ! -e "/sys/class/net/${nic}" ]]; then
  echo "G1 network: interface ${nic} does not exist" >&2
  exit 2
fi

if [[ "$(<"/sys/class/net/${nic}/carrier")" != "1" ]]; then
  echo "G1 network: ${nic} has no cable carrier" >&2
  exit 3
fi

current_address="$(ip -4 -o addr show dev "${nic}" | awk '{print $4}')"
if [[ " ${current_address} " != *" ${expected_address} "* ]]; then
  nmcli connection up uuid "${connection_uuid}" ifname "${nic}" >/dev/null
fi

current_address="$(ip -4 -o addr show dev "${nic}" | awk '{print $4}')"
if [[ " ${current_address} " != *" ${expected_address} "* ]]; then
  echo "G1 network: expected ${expected_address} on ${nic}, got ${current_address:-none}" >&2
  exit 4
fi

route_device="$(ip route get 192.168.123.161 | awk '/dev/ {for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
if [[ "${route_device}" != "${nic}" ]]; then
  echo "G1 network: route to 192.168.123.161 uses ${route_device:-no interface}" >&2
  exit 5
fi

echo "G1 network ready: ${nic} ${expected_address}; robot subnet route is active"
