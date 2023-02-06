#/bin/bash 
# Go to previous playing or next song or pause using `playerctl`,
# and send a notification using `notify-send`

icon="/usr/share/icons/ePapirus-Dark/24x24/apps/spotify.svg"

case $1 in

    prev)
        title=$(playerctl metadata title);
        artist_album=$(playerctl metadata --format "{{ artist }} ({{ album }})");
        playerctl previous;
        notify-send "play previous" -i $icon;
        ;;

    next)
        playerctl next;
        if [ $(playerctl status)=="Playing" ]; then
            title=$(playerctl metadata title);
            artist_album=$(playerctl metadata --format "{{ artist }} ({{ album }})");
            notify-send "play next" "$title" -i $icon;
        else
            sleep 1s;
            title=$(playerctl metadata title);
            artist_album=$(playerctl metadata --format "{{ artist }} ({{ album }})");
            notify-send "play next" "$title" -i $icon;
        fi
        ;;

    pause)
        title=$(playerctl metadata title);
        artist_album=$(playerctl metadata --format "{{ artist }} ({{ album }})");
        playerctl play-pause;
        notify-send "toggle song" -i $icon;
        ;;

    *)
        echo "Unknow option."
        echo "Usage: songctl <prev|pause|next>"
        ;;
esac

