-- local colorscheme = 'tokyonight-night'
local colorscheme = 'tokyonight-moon'
-- local colorscheme = 'tokyonight-storm'
-- local colorscheme = 'tokyonight-day'
-- local colorscheme = 'github_dark_dimmed'

local is_ok, _= pcall(vim.cmd,"colorscheme " .. colorscheme)
if not is_ok then
    vim.notify('colorscheme ' .. colorscheme .. ' not found!')
    return
end
