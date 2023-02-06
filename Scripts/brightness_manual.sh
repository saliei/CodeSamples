#!/usr/bin/env bash

backlight_dir="/sys/class/backlight/amdgpu_bl0";
max_brightness=$(cat "$backlight_dir/max_brightness");

function get_brightness
{
    cat "$backlight_dir/brightness";
}

function brightness_up
{
    current_brightness=$(get_brightness)
    inc10=$(( $max_brightness * 5 / 100 ))
    new_brightness=$(( $current_brightness + $inc10 ))
    if [ $new_brightness -le $max_brightness ]; then
        echo $new_brightness > "$backlight_dir/brightness"
    else
        echo $max_brightness > "$backlight_dir/brightness"
    fi
    
}

function brightness_dec
{
    current_brightness=$(get_brightness)
    dec10=$(( $max_brightness * 5 / 100 ))
    new_brightness=$(( $current_brightness - $dec10 ))
    if [ $new_brightness -le 0 ]; then
        echo 10 > "$backlight_dir/brightness"
    else
        echo $new_brightness > "$backlight_dir/brightness"
    fi
}

function send_notification
{
  brightness=$(get_brightness)
  brightness_scaled=$(( $brightness * 100 / $max_brightness ))
  if [[ $brightness_scaled -ge 80  ]]; then
      icon="/usr/share/icons/Papirus-Dark/symbolic/status/display-brightness-high-symbolic.svg"
  elif [[ $brightness -ge 40 && $brightness -lt 80 ]]; then
      icon="/usr/share/icons/Papirus-Dark/symbolic/status/display-brightness-medium-symbolic.svg"
  else
      icon="/usr/share/icons/Papirus-Dark/symbolic/status/display-brightness-low-symbolic.svg"
  fi
  # Make the bar with the special character ─ (it's not dash -)
  # https://en.wikipedia.org/wiki/Box-drawing_character
  bar=$(seq -s "─" 0 $((brightness / 12)) | sed 's/[0-9]//g')
  # Send the notification
  dunstify -i "$icon" -r 5555 -u normal "  $bar $brightness_scaled%"
}




case $1 in
    up)
        brightness_up
        send_notification
        ;;
    down)
        brightness_dec
        send_notification
        ;;
esac
