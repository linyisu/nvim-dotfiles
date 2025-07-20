if vim.g.vscode then
    -- VSCode-neovim 配置（如有）
else
    require('options')
    require('lazy-init')
    require('colorscheme')
    require('keymaps')
end
