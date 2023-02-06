#!/usr/bin/sh

# primary display
prm="eDP"
# secondary display
sec="HDMI-A-0"
# secondary display size
sec_size="1920x1080"

selection=$( \
	echo -e "Primary\nSecondary\nMirror\nExtend" |\
	rofi -dmenu -i -p 'Option');
echo $selection;

sleep .2

case $selection in
	Primary)
		xrandr --output $prm --primary --output $sec --off
		;;
	Secondary)
		xrandr --output $prm --off --output $sec --mode $sec_size --primary
		;;
	Mirror)
		xrandr --output $prm --auto --output $sec --mode $sec_size --same-as $prm
		;;
	Extend)
		xrandr --output $prm --auto --primary --output $sec --mode $sec_size --left-of $prm
		;;
esac
