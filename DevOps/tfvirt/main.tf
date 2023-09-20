terraform {
  required_providers {
    libvirt = {
      source = "dmacvicar/libvirt"
      version = "0.7.1"
    }
  } 
}

provider "libvirt" {
  uri = "qemu:///system"
}

variable "VM_COUNT" {
  default = 3
  type = number
}

variable "VM_USER" {
  default = "saltf"
  type = string
}

variable "VM_HOSTNAME" {
  default = "vm"
  type = string
}

variable "VM_IMG_URL" {
  default = "/home/saliei/downs/CentOS-Stream-GenericCloud-9-latest.x86_64.qcow2"
  type = string
}

variable "VM_IMG_FORMAT" {
  default = "qcow2"
  type = string
}

variable "VM_CIDR_RANGE" {
  default = "10.10.10.10/24"
  type = string
}

data "template_file" "user_data" {
  template = file("${path.module}/cloud_init.cfg")
  vars = {
    VM_USER = var.VM_USER
  }
}

data "template_file" "network_config" {
  template = file("${path.module}/network_config.cfg")
}

resource "libvirt_pool" "vm" {
  name = "${var.VM_HOSTNAME}_pool"
  type = "dir"
  path = "tmp/terraform-provider-libvirt-pool-centos"
}

resource "libvirt_volume" "vm" {
  count = var.VM_COUNT
  name = "${var.VM_HOSTNAME}-${count.index}_volume.${var.VM_IMG_FORMAT}"
  pool = libvirt_pool.vm.name
  source = var.VM_IMG_URL
  format = var.VM_IMG_FORMAT
}

resource "libvirt_network" "vm_public_network" {
  name = "${var.VM_HOSTNAME}_network"
  mode = "nat"
  domain = "${var.VM_HOSTNAME}.local"
  addresses = ["${var.VM_CIDR_RANGE}"]
  dhcp {
    enabled = true
  }
  dns {
    enabled = true
  }
}

resource "libvirt_cloudinit_disk" "cloudinit" {
  name = "${var.VM_HOSTNAME}_cloudinit.iso"
  #name = "${var.VM_IMG_URL}"
}

resource "libvirt_domain" "vm" {
  count = var.VM_COUNT
  name = "${var.VM_HOSTNAME}-${count.index}"
  memory = "1024"
  vcpu = 1

  cloudinit = "${libvirt_cloudinit_disk.cloudinit.id}"

  network_interface {
    network_id = "${libvirt_network.vm_public_network.id}"
    network_name = "${libvirt_network.vm_public_network.name}"
  }

  console {
    type = "pty"
    target_type = "serial"
    target_port = "0"
  }

  console {
    type = "pty"
    target_type = "virtio"
    target_port = "1"
  }

  disk {
    volume_id = "${libvirt_volume.vm[count.index].id}"
  }

  graphics {
    type = "spice"
    listen_type = "address"
    autoport = true
  }
}
