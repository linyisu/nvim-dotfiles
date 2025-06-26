if vim.g.vscode then
    -- VSCode-neovim 配置（如有）
else
    require('options')
    require('keymaps')
    require('lazy-init')
    require('colorscheme')
end
