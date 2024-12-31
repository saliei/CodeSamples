vim.g.base46_cache = vim.fn.stdpath "data" .. "/base46/"
vim.g.mapleader = " "

-- bootstrap lazy and all plugins
local lazypath = vim.fn.stdpath "data" .. "/lazy/lazy.nvim"

if not vim.uv.fs_stat(lazypath) then
    local repo = "https://github.com/folke/lazy.nvim.git"
    vim.fn.system {
        "git",
        "clone",
        "--filter=blob:none",
        repo,
        "--branch=stable",
        lazypath,
    }
end

vim.opt.rtp:prepend(lazypath)

local lazy_config = require "configs.lazy"
-- load plugins
require("lazy").setup({
    {
        "NvChad/NvChad",
        lazy = false,
        branch = "v2.5",
        import = "nvchad.plugins",
    },

    { import = "plugins" },
}, lazy_config)

dofile(vim.g.base46_cache .. "defaults")
dofile(vim.g.base46_cache .. "statusline")

require "options"
require "nvchad.autocmds"

vim.schedule(function()
    require "mappings"
end)

require("nvim-tree").setup {
    view = { adaptive_size = true },
}

require("nvim-lastplace").setup()

local cmp = require "cmp"
cmp.setup {
    preselect = cmp.PreselectMode.None,
    completion = {
        completeopt = "menu,preview,menuone,noinsert,noselect",
    },
    mapping = cmp.mapping.preset.insert {
        ["<CR>"] = cmp.mapping.confirm { select = false },
    },
}

vim.g.disable_autoformat = true
