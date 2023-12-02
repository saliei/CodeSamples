#!/usr/bin/env
#
# Install Arch Linux. Assumes the Live image is booted 
# and has internet access

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

CURRENT_DIR="$(basename $0)"

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

function partition() {

}

function main() {
  echo "this is main function"
}

main "$@"
