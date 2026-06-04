# Neovim Configuration

这是一套基于 LazyVim starter 的 Neovim 配置。配置保留了前面已经调整过的习惯：`mini.files` 文件管理器、居中的文件/文本搜索、C/C++/Rust 运行命令、C/C++ 格式化、Neovide 字体设置和自定义首页。

## 系统要求

### 必须提前安装

这些是第一次启动前必须已经存在于系统中的软件：

| 依赖 | Windows | Linux | 说明 |
| --- | --- | --- | --- |
| Neovim | 必须 | 必须 | LazyVim 要求 Neovim `0.11.2` 或更新版本，并且需要 LuaJIT。 |
| Git | 必须 | 必须 | 用于第一次启动时从 GitHub 下载 `lazy.nvim`、LazyVim 和插件。建议 Git `2.19.0` 或更新版本。 |
| 网络连接 | 必须 | 必须 | 第一次启动需要访问 GitHub；LSP 工具会通过 Mason 下载。 |

### 建议提前安装

这些不是 Neovim 启动的硬性条件，但会影响体验或某些功能：

| 功能 | 建议安装 | 说明 |
| --- | --- | --- |
| 图标显示 | Nerd Font | LazyVim、首页、文件图标会用到 Nerd Font。Neovide 可通过 `NEOVIDE_FONT` 环境变量覆盖字体。 |
| Treesitter 高亮 | C 编译器 | LazyVim 的 Treesitter parser 安装可能需要 C 编译器。Windows 可用 Visual Studio Build Tools / LLVM / MinGW，Linux 可用系统包管理器安装 gcc/clang。 |
| 运行 C/C++ 当前文件 | `g++` 或 `clang++`，C 文件可用 `gcc` 或 `clang` | `空格 rr` / `:RunFile` 需要真实编译器。`clangd` 只是 LSP，不等于 C++ 编译器。 |
| 运行 Rust 项目 | Rust 工具链中的 `cargo` | 在 Rust 文件里按 `空格 rr` 会向上查找最近的 `Cargo.toml`，并在项目根目录执行 `cargo run`。 |
| Linux 系统剪贴板 | `wl-clipboard` 或 `xclip` / `xsel` | 没有这些工具时，配置不会强行启用系统剪贴板。Windows 通常不需要额外安装。 |

## 会自动下载的内容

以下内容不需要手动提前安装：

| 组件 | 安装方式 |
| --- | --- |
| `lazy.nvim` | 第一次启动 Neovim 时由本配置自动从 GitHub 克隆。 |
| LazyVim 和插件 | 由 `lazy.nvim` 自动下载和管理。 |
| Lua LSP (`lua_ls`) | 由 LazyVim + Mason 自动安装。 |
| C/C++ LSP (`clangd`) | 由 LazyVim + Mason 自动安装。 |
| Rust LSP (`rust-analyzer`) | 由 LazyVim + Mason 自动安装。 |
| C/C++ 格式化器 (`clang-format`) | 由 Mason 自动安装；C/C++ 格式化会读取项目根目录里的 `.clang-format`，该文件需要自行添加。 |
| Treesitter 解析器 | `cpp`、`rust`、`toml` 等解析器由 Treesitter 自动安装。 |
| LuaSnip | 由 `lazy.nvim` 自动安装；C++ 竞赛 snippets 已内置，不需要安装 VSCode 或额外 snippet 包。 |

建议在每个 C/C++ 项目根目录手动添加 `.clang-format`。本配置不会在打开文件时自动生成该文件；需要时可以在项目根目录执行 `:ClangFormatInit` 生成下面这份默认模板，或执行 `:ClangFormatOpen` 打开当前 workspace 根目录的 `.clang-format`。

建议 `.clang-format` 内容：

```yaml
BasedOnStyle: Google
Standard: Latest
IndentWidth: 4
ColumnLimit: 120
AccessModifierOffset: -4
InsertBraces: true
```

其中 `InsertBraces: true` 会让支持该选项的 `clang-format` 在格式化时给无大括号的 `if`、`for`、`while` 等语句补上大括号。

## 配置目录

把本仓库内容放到对应目录后，直接启动 `nvim` 即可：

| 系统 | 配置目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\nvim` |
| Linux | `~/.config/nvim` |

## 常用按键

| 按键 | 功能 |
| --- | --- |
| `空格 e` | 打开/关闭文件管理器。 |
| `空格 ff` | 在当前 workspace 中查找文件。 |
| `空格 fg` | 在当前 workspace 中搜索文本。 |
| `空格 rn` | 重命名光标下的符号，需要当前文件有可用 LSP。 |
| `空格 rr` | 编译并运行当前 C/C++ 文件；Rust 文件会在 Cargo 项目根目录执行 `cargo run`。 |
| `空格 H` | 打开首页。 |
| `Alt + 上/下` | Normal 模式移动当前行；Visual 模式移动选中行并保持选区。 |
| `空格 ?` | 查看当前 buffer 可用快捷键。 |

`空格 rr` 打开的运行窗口默认停在 Vim 模式，可以直接移动、复制和搜索输出；按 `i` 进入程序输入模式，按 `Esc` 回到 Vim 模式，按 `q` 关闭运行窗口。

文件管理器中：

| 按键 | 功能 |
| --- | --- |
| `Enter` / `右方向键` | 进入目录或打开文件。 |
| `Backspace` / `-` / `左方向键` | 返回上级目录。 |
| `j` / `k` 或上下方向键 | 上下移动，首尾循环。 |
| `a` | 新建文件。 |
| `A` | 新建文件夹。 |
| `r` | 重命名。 |
| `d` | 删除。 |
| `R` | 刷新。 |
| `o` | 把当前选中的目录作为 workspace。 |
| `gh` | 跳到家目录。 |
| `gc` | 跳到 Neovim 配置目录。 |
| `gw` | 跳到当前 workspace。 |
| `q` 或 `:q` | 关闭文件管理器窗口。 |

## 检查命令

进入 Neovim 后可以用这些命令检查状态：

```vim
:Lazy
:Mason
:checkhealth
:checkhealth vim.lsp
```

如果 C/C++ 或 Rust 高亮、LSP、诊断、运行命令不可用，优先检查 `:Lazy` 和 `:Mason` 是否完成安装。C/C++ 运行需要系统里有真实编译器；Rust 运行需要系统里有 `cargo`，并且当前文件上级目录中存在 `Cargo.toml`。

## 测试流程

本配置带有一个纯 Lua headless 测试入口，不依赖 `busted`、`plenary.nvim` 等额外测试框架：

```sh
nvim --headless -i NONE -l tests/run.lua
```

维护规则：

1. 每次新增配置代码，都要同时新增或更新对应测试。
2. 每次修改配置代码，都要先补上能暴露该行为的测试，再修改实现。
3. 每次改完必须运行完整测试命令，确认全部通过后再提交。
4. 涉及 Windows/Linux 差异时，测试应避免硬编码路径分隔符，优先使用 `vim.fs.normalize()` 或 `vim.fs.joinpath()`。

测试文件放在 `tests/specs/*_spec.lua`，`tests/run.lua` 会自动发现并运行它们。

注意：测试入口会加载正常的 LazyVim 配置和插件管理器。第一次在新机器运行测试时，仍然需要满足本 README 前面列出的基础要求，并可能需要先完成插件下载。
