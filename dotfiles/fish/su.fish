# Redirect users shell to fish when starting with su
function su
    command su --shell=/usr/bin/fish $argv
end
