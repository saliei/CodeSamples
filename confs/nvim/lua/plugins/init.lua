local plugins = {
    {
        "stevearc/conform.nvim",
        -- event = "BufWritePre",
        opts = require "configs.conform",
    },

    {
        "neovim/nvim-lspconfig",
        opts = {
            servers = {
                clangd = {
                    mason = false,
                },
            },
        },
        config = function()
            require "configs.lspconfig"
        end,
    },

    {
        "nvim-treesitter/nvim-treesitter",
        config = function()
            require "configs.treesitter"
        end,
    },

    {
        "mfussenegger/nvim-lint",
        -- event = { "BufReadPre", "BufNewFile" },
        config = function()
            require "configs.lint"
        end,
    },

    {
        "zapling/mason-conform.nvim",
        event = "VeryLazy",
        dependencies = { "conform.nvim" },
        config = function()
            require "configs.mason-conform"
        end,
    },

    {
        "williamboman/mason-lspconfig.nvim",
        event = "VeryLazy",
        dependencies = { "nvim-lspconfig" },
        config = function()
            require "configs.mason-lspconfig"
        end,
    },

    {
        "rshkarin/mason-nvim-lint",
        event = "VeryLazy",
        dependencies = { "nvim-lint" },
        config = function()
            require "configs.mason-lint"
        end,
    },

    {
        "ethanholz/nvim-lastplace",
        event = "BufWinEnter",
    },
}

return plugins
