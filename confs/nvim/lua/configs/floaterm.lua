local m = {}

m.setup = function ()
    vim.g.floaterm_width = 0.8
    vim.g.floaterm_height = 0.8
    vim.g.floaterm_opener = "vsplit"
    vim.g.floaterm_position = "center"
end

-- Execute setup when this file is loaded
m.setup()
return m
