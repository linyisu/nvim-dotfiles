-- -------------------------------------------------------
-- 剪贴板与补全 (Clipboard & Completion)
-- -------------------------------------------------------
vim.opt.clipboard = 'unnamedplus' -- 与系统剪贴板共享内容 (实现 Neovim 内外复制粘贴)
vim.opt.completeopt = { 'menu', 'menuone', 'noselect' } -- 自动补全菜单的选项：显示菜单，只有一个候选项时也显示，不自动选择第一项

-- -------------------------------------------------------
-- 编辑器行为 (Editor Behavior)
-- -------------------------------------------------------
vim.opt.mouse = 'a'              -- 在所有模式下启用鼠标
vim.opt.wrap = true              -- 启用自动换行 (如果想关闭，可以设为 false)
vim.opt.showmode = true          -- 在底部显示当前模式 (如 -- INSERT --)

-- -------------------------------------------------------
-- 缩进与制表符 (Indentation & Tabs)
-- -------------------------------------------------------
vim.opt.tabstop = 4              -- 一个 Tab 键在视觉上代表的空格数
vim.opt.softtabstop = 4          -- 在编辑时，按一次 Tab 键实际插入的空格数
vim.opt.shiftwidth = 4           -- 执行缩进操作 (如 `>>` 或 `<<`) 时，移动的空格数
vim.opt.expandtab = true         -- 将 Tab 键自动转换为空格 (对 Python 等语言非常重要)

-- -------------------------------------------------------
-- 界面显示 (UI Display)
-- -------------------------------------------------------
vim.opt.termguicolors = true     -- 启用 24-bit 真彩色，让主题颜色正确显示
vim.opt.number = true            -- 显示绝对行号
vim.opt.relativenumber = true    -- 显示相对行号 (方便使用 `j` `k` 进行跳转)
vim.opt.cursorline = true        -- 高亮显示光标所在的行
vim.opt.splitbelow = true        -- 创建水平分屏时，新窗口显示在下方
vim.opt.splitright = true        -- 创建垂直分屏时，新窗口显示在右侧
vim.opt.signcolumn = "yes"       -- 始终显示符号列 (用于显示 Git 状态、LSP 诊断等图标)

-- -------------------------------------------------------
-- 搜索设置 (Searching)
-- -------------------------------------------------------
vim.opt.incsearch = true         -- 在输入搜索内容时，实时高亮匹配结果
vim.opt.hlsearch = false         -- 默认关闭搜索结果的持续高亮 (可以用 `:noh` 临时关闭)
vim.opt.ignorecase = true        -- 搜索时默认忽略大小写
vim.opt.smartcase = true         -- 如果搜索内容中包含大写字母，则切换为大小写敏感搜索

-- -------------------------------------------------------
-- Neovide (图形化客户端) 专属配置
-- -------------------------------------------------------
-- 仅当检测到当前环境是 Neovide 时，以下配置才会生效
if vim.g.neovide then
    vim.g.neovide_opacity = 0.80                   -- 设置窗口透明度 (0.0 完全透明, 1.0 完全不透明)
    vim.g.neovide_confirm_quit = true              -- 关闭 Neovide 时弹出确认提示
    vim.g.neovide_hide_mouse_when_typing = true    -- 打字时自动隐藏鼠标指针

    -- 光标动画效果 (vfx = visual effects)
    vim.g.neovide_cursor_vfx_mode = ""             -- 关闭光标动画 (可选值: "railgun", "torpedo", "pixiedust", "sonicboom", "ripple", "wireframe")
    vim.g.neovide_cursor_trail_length = 0.03       -- 光标拖尾效果的长度

    -- 窗口内边距
    vim.g.neovide_padding_top = 0
    vim.g.neovide_padding_bottom = 0
    vim.g.neovide_padding_right = 0
    vim.g.neovide_padding_left = 0

    -- 设置 Neovide 使用的字体和大小
    vim.o.guifont = "JetBrainsMono Nerd Font:h14"
end
