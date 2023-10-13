#!/usr/bin/bash
# 
# Various utility snippets to work with libvirt and virsh

virt-install \
    --name testvm \
    --ram 2048 \
    --disk path=/var/lib/libvirt/images/ubuntu-server-22.04_test.qcow2,size=8 \
    --vcpu 2 \
    --os-variant generic \
    --console pty,target_type=serial \
    --bridge=br0 \
    --cdrom $HOME/downs/iso/ubuntu-22.04.3-live-server-amd64.iso

