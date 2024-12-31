require "nvchad.mappings"

local cmd = vim.api.nvim_create_user_command
local map = vim.keymap.set

cmd("FormatDisable", function(args)
    if args.bang then
        vim.b.disable_autoformat = true
    else
        vim.g.disable_autoformat = true
    end
end, { desc = "Disable autoformat-on-save", bang = true })

cmd("FormatEnable", function()
    vim.b.disable_autoformat = false
    vim.g.disable_autoformat = false
end, { desc = "Re-enable autoformat-on-save" })

map("n", ";", ":", { desc = "CMD enter command mode" })
map("i", "jk", "<ESC>")

map("n", "<leader>td", function()
    vim.diagnostic.enable(not vim.diagnostic.is_enabled())
end, { desc= "toggle diagnostic" ,silent = true, noremap = true })

map("", "<leader>fc", function()
    require("conform").format { async = true, lsp_fallback = true }
end, { desc = "format" })

map("n", "<leader>ft", function()
    if vim.b.disable_autoformat or vim.g.disable_autoformat then
        vim.cmd "FormatEnable"
    else
        vim.cmd "FormatDisable"
    end
end, { desc = "toggle format" })
