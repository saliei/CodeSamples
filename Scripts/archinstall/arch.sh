#!/usr/bin/env bash
#
# Install Arch Linux. Assumes the Live image is booted 
# and has internet access

DISK_PARTITION="sda"
ROOT_PARTITION="8GB"
HOME_PARTITION="6GB"
SWAP_PARTITION="2GB"

REGION="Europe"
CITY="Rome"
HOSTNAME="darkstar"
USERNAME="saliei"

CURRENT_DIR="$(basename $0)"

BLACK="\033[0;30m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
PURPLE="\033[0;35m"
CYAN="\033[0;36m"
WHITE="\033[0;37m"
BGREEN="\033[1;32m"
BRED="\033[1;31m"
RESET="\033[0m"

trap "DIE" SIGINT
trap "DIE" SIGQUIT

function DIE() {
    echo -e "\nsetup.sh: exiting on interrupt..."
    echo "removing any temporary files"
    exit 127
}

function LOG() {
    CURDATE="${BLUE}$(date +'%Y-%m-%d %T')${RESET}"
    LOGGER="setup"
    ORIG_LOGLEVEL="$1"

    case $1 in
        "DEBUG")
            shift
            LOGLEVEL="${GREEN}DEBUG${RESET}"
            ;;
        "INFO")
            shift
            LOGLEVEL="${CYAN} INFO${RESET}"
            ;;
        "WARN")
            shift
            LOGLEVEL="${YELLOW} WARN${RESET}"
            ;;
        "ERROR")
            shift
            LOGLEVEL="${RED}ERROR${RESET}"
            ;;
        "FATAL")
            shift
            LOGLEVEL="${BRED}FATAL${RESET}"
            ;;
        *)
            LOGLEVEL="${WHITE}NOLEVEL${RESET}"
            ;;
    esac

    LOGMSG="$1"
    echo -e "$CURDATE $LOGGER $LOGLEVEL: $LOGMSG"
    #[[ "$ORIG_LOGLEVEL" == "FATAL" ]] && DIE
}

function _has_internet_access() {
    echo "this checks the network access"
}

#TODO: more flexibility on partitioning schema
function _partition() {
    LOG DEBUG "partitioning"
    _disk_line=$(sfdisk -l | grep "Disk /dev/${DISK_PARTITION}" &>/dev/null)
    LOG INFO "sfdisk info: ${_disk_line}"
    LOG WARN "wiping all signatures on disk: /dev/${DISK_PARTITION}"
    wipefs --all /dev/${DISK_PARTITION}

    _hw_sector_size=$(cat /sys/block/${DISK_PARTITION}/queue/hw_sector_size)
    LOG INFO "using hw sector size: ${_hw_sector_size}"
    sfdisk /dev/${DISK_PARTITION} < "${CURRENT_DIR}/files/${DISK_PARTITION}.sfdisk"
    [[ $? != 0 ]] && LOG ERROR "problem in partitioning the disk: /dev/${DISK_PARTITION}"

    LOG WARN "formatting disk: /dev/${DISK_PARTITION}1"
    mkfs.fat -F 32 /dev/${DISK_PARTITION}1
    LOG WARN "formatting disk: /dev/${DISK_PARTITION}2"
    mkswap /dev/${DISK_PARTITION}2
    LOG WARN "formatting disk: /dev/${DISK_PARTITION}3"
    mkfs.ext4 /dev/${DISK_PARTITION}3
}

# TODO: add other user-space packages, e.g. networking
function pre_installation() {
    _partition

    LOG DEBUG "mounting EFI partition: /dev/${DISK_PARTITION}1 /mnt/boot"
    mount --mkdir /dev/${DISK_PARTITION}1 /mnt/boot/efi
    LOG DEBUG "swap on: /dev/${DISK_PARTITION}2"
    swapon /dev/${DISK_PARTITION}2
    LOG DEBUG "mounting root partition: /dev/${DISK_PARTITION}3 /mnt"
    mount --mkdir /dev/${DISK_PARTITION}3 /mnt
    LOG INFO "installing essential packages"
    pacstrap -K /mnt base linux linux-firmware
}

#TODO: set systemd-timesyncd
#TODO: check permissions as this is run in the script
function configurations() {
    LOG WATN "generating fstab"
    genfstab -L /mnt >> /mnt/etc/fstab
    LOG WARN "setting localtime to: ${REGION}/${CITY}"
    arch-chroot /mnt ln -sf /usr/share/zoneinfo/${REGION}/${CITY} /etc/localtime
    LOG INFO "setting hwclock to system"
    arch-chroot /mnt hwclock --systohc
    LOG DEBUG "uncommenting en_US.UTF-8 UTF-8 locale"
    sed -i "/en_US.UTF-8 UTF-8/s/^#*//g" /mnt/etc/locale.gen 
    LOG INFO "generating locale"
    arch-chroot /mnt locale-gen
    LOG INFO "copying locale.conf"
    cp "${CURRENT_DIR}/files/locale.conf" /mnt/etc/
    LOG INFO "copying vconsole.conf"
    cp "${CURRENT_DIR}/files/vconsole.conf" /mnt/etc/
    LOG INFO "setting hostname to: ${HOSTNAME}"
    echo "${HOSTNAME}" > /mnt/etc/hostname

    #TODO: network config

    read -s -p "enter root password:" _root_pass1
    read -s -p "enter root password again:" _root_pass2
    while [[ "${_root_pass1}" != "${_root_pass2}" ]]; do
        LOG ERROR "passwords didn't match, prompting again"
        read -s -p "enter root password:" _root_pass1
        read -s -p "enter root password again:" _root_pass2
    done
    _root_pass=$_root_pass1
    LOG WARN "changing root password"
    echo "root:$_root_pass" | chpasswd --root /mnt
    LOG WARN "adding user: ${USERNAME}"
    useradd --root /mnt -c "Saeid Aliei" -m "${USERNAME}"
    LOG WANT "using root password also for user: ${USERNAME}"
    echo "${USERNAME}:$_root_pass" | chapsswd --root /mnt
}


function main() {
    pre_installation
    configurations
}

main "$@"
