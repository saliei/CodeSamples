 set -gx EDITOR (type -p vim)
 set -gx BROWSER (type -p google-chrome-stable)

fish_add_path -p ~/.local/bin

function su
    command su --shell=/usr/bin/fish $argv
end

function e
    exit
end

function gh
    cd $HOME/Documents/github/
end

function nt
    notes $argv
end

function penv
    source $HOME/.penvs/$argv/bin/activate.fish
end

function gssh
    eval (ssh-agent -c)
    ssh-add ~/.ssh/github
end

function vpn
    sudo geph4-client connect --username "salad123" --password "123456789" --vpn-mode tun-route $argv
end
