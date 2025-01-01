local m = {}

m.setup = function ()
    local lazygit = require("lazygit")
    -- Transparency of floating window
    vim.g.lazygit_floating_window_winblend = 0
    vim.g.lazygit_floating_window_scaling_factor = 0.9
    vim.g.lazygit_use_neovim_remote = 1
end

m.setup()
return m
