function github
    if count $argv > /dev/null
        cd ~/Documents/github/$argv
    else
        cd ~/Documents/github/
    end
end
