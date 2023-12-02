#!/usr/bin/env
#
# Install Arch Linux. Assumes the Live image is booted 
# and has internet access

DISK_PARTITION="sda"
ROOT_PARTITION="8GB"
HOME_PARTITION="6GB"
SWAP_PARTITION="2GB"

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

function has_internet_access() {
    echo "this checks the network access"
}

function partition() {
    LOG DEBUG "partitioning"
    _disk_line=$(sfdisk -l | grep "Disk /dev/${DISK_PARTITION}" &>/dev/null)
    LOG INFO "sfdisk info: ${_disk_line}"
    LOG WARN "wiping all signatures on disk: /dev/${DISK_PARTITION}"
    wipefs --all /dev/${DISK_PARTITION}

    _hw_sector_size=$(cat /sys/block/${DISK_PARTITION}/queue/hw_sector_size)

}

function main() {
  echo "this is main function"
}

main "$@"
