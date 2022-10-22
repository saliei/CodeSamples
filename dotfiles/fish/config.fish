 set -gx EDITOR (type -p vim)
 set -gx VISUAL (type -p vim)
 set -gx BROWSER (type -p google-chrome-stable)

fish_add_path -p ~/.local/bin

# Redirect users shell to fish when starting with su
function su
    command su --shell=/usr/bin/fish $argv
end

function q
    exit
end

function gh
    cd $HOME/Documents/github/
end

function pr
    cd $HOME/Documents/projects/
end

function sc
    cd $HOME/Documents/scratch/
end

function nt
    notes $argv
end

function penv
    source $HOME/Documents/penvs/$argv/bin/activate.fish
end

function lh
    ls -lash $argv
end

function ssh_add
    if ! pgrep ssh-agent > /dev/null
        eval (ssh-agent -c)
    end
    ssh-add ~/.ssh/id_rsa_github_manjaro_i3
end

