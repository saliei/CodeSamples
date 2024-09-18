local options = {
    ensure_installed = {
        "bash",
        "fish",
        "lua",
        "luadoc",
        "markdown",
        "printf",
        "vim",
        "vimdoc",
        "toml",
        "yaml",
        "cuda",
        "python",
        "go",
        -- "gomod",
        -- "gosum",
        -- "gotmpl",
        -- "gowork",
        "c",
        "cpp",
        "cmake",
        "make",
    },

    highlight = {
        enable = true,
        use_languagetree = true,
    },

    indent = { enable = true },
}

require("nvim-treesitter.configs").setup(options)
