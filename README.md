# nvim-dotfiles

基于 AstroNvim v5 的个人 Neovim 配置。

这套配置面向日常开发和通用文本编辑，兼顾 C++ / Python / Rust 工作流。整体尽量保留 AstroNvim 的默认行为，只对补全、LSP、代码片段、终端工作流和 health 输出做少量定制。

## 依赖

- Neovim 0.10 或更新版本
- Git
- ripgrep (`rg`)
- fd
- Nerd Font，本配置中使用的是 `JetBrainsMono Nerd Font`
- 可选但推荐安装：`lazygit`、`node`、`python`、`cargo`、`tree-sitter`

大部分语言工具会在启动时通过 Mason 自动安装：

- `clangd`
- `clang-format`
- `rust-analyzer`
- `lua-language-server`
- `json-lsp`
- `black`
- `isort`
- `ruff`
- `stylua`
- `tree-sitter-cli`，当系统中没有 `tree-sitter` 命令时安装

## 安装

先备份已有的 Neovim 配置：

```sh
mv ~/.config/nvim ~/.config/nvim.bak
mv ~/.local/share/nvim ~/.local/share/nvim.bak
mv ~/.local/state/nvim ~/.local/state/nvim.bak
mv ~/.cache/nvim ~/.cache/nvim.bak
```

克隆本仓库：

```sh
git clone git@github.com:linyisu/nvim-dotfiles.git ~/.config/nvim
```

如果没有配置 SSH，也可以使用 HTTPS：

```sh
git clone https://github.com/linyisu/nvim-dotfiles.git ~/.config/nvim
```

启动 Neovim：

```sh
nvim
```

首次启动时，Lazy.nvim 和 Mason 会自动安装缺失的插件与工具。启动完成后，可以用 `:Lazy sync` 和 `:Mason` 检查安装状态。

## 主要功能

- 基于 AstroNvim v5 和 `lazy.nvim` 的插件结构
- 使用 Tokyonight Moon 主题
- 使用 `blink.cmp` 作为补全框架，支持 `<Tab>` / `<S-Tab>` 导航
- Treesitter 高亮和 textobjects
- clangd 和 `clangd_extensions.nvim`
- 通过 `rustaceanvim` 提供 Rust 支持
- 通过 Mason 管理 Python 格式化工具
- 通过 LuaSnip 提供自定义 C++ 代码片段
- 自动保存，并保留正常的写入 autocmd
- 非 Windows 环境下 shell 跟随 `$SHELL`，不会写死为某个 shell
- 过滤已知的、无实际影响的 health 提示

## 快捷键

Leader 是 `<Space>`。

| 快捷键 | 作用 |
| --- | --- |
| `<C-x>` | 下一个 buffer |
| `<C-z>` | 上一个 buffer |
| `<Leader>bd` | 选择并关闭 buffer |
| `<Leader>m` | 打开 zoxide picker |
| `<Leader>rn` | 通过 LSP 重命名符号 |
| `<Leader>Tf` | 打开浮动终端 |
| `<Leader>Th` | 打开水平终端 |
| `<Leader>Tv` | 打开垂直终端 |
| `<Leader>Tg` | 打开 lazygit |
| `<A-k>` / `<A-j>` | 通过 `mini.move` 移动当前行或选区 |
| `<A-Up>` / `<A-Down>` | `<A-k>` / `<A-j>` 的别名 |
| `<C-Up>` / `<C-Down>` | 调整窗口高度 |
| `<C-Left>` / `<C-Right>` | 调整窗口宽度 |

## 仓库结构

```text
init.lua
lua/
  community.lua
  compat.lua
  health_filters.lua
  keymaps.lua
  lazy_setup.lua
  polish.lua
  plugins/
```

- `lua/community.lua`：AstroCommunity 导入
- `lua/lazy_setup.lua`：AstroNvim 和 lazy.nvim 启动配置
- `lua/plugins/`：各插件的独立配置
- `lua/keymaps.lua`：AstroCore 之外的用户快捷键
- `lua/polish.lua`：最后执行的启动设置和自动保存
- `lua/compat.lua`：当前 Neovim / 插件 API 变化的兼容层
- `lua/health_filters.lua`：过滤已知的 health 噪音提示

## 维护

常用命令：

```sh
nvim --headless "+Lazy! sync" +qa
nvim --headless "+checkhealth" +qa
find lua -name '*.lua' -exec luac -p {} +
```
