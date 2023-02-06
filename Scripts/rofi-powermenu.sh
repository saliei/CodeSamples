#!/usr/bin/sh
selection=$( \
	echo -e "Lock\nLogout\nSuspend\nHibernate\nReboot\nShutdown" |\
	rofi -dmenu -i -p 'Option');
echo $selection;

sleep .2

case $selection in
	Lock)
		i3exit lock
		;;
	Logout)
		i3exit logout
		;;
	Suspend)
		i3exit suspend
		;;
	Hibernate)
		i3exit hibernate
		;;
	Reboot)
		i3exit reboot
		;;
	Shutdown)
		i3exit shutdown
		;;
esac
