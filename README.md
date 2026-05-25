# Neovim Configuration

这是一套从零配置的 Neovim 配置，不基于任何发行版式框架。配置目标是保持结构清晰、可迁移，并尽量让插件、LSP 等组件在首次打开 Neovim 时自动从 GitHub 或 Mason 下载。

## 系统要求

### 必须提前安装

以下软件需要在使用本配置前先安装好：

| 依赖 | Windows | Linux | 说明 |
| --- | --- | --- | --- |
| Neovim | 必须 | 必须 | 建议使用 Neovim `0.12` 或更新版本。本配置已在 `0.12.2` 上验证。 |
| Git | 必须 | 必须 | 用于首次自动下载 `lazy.nvim` 和后续插件。 |
| 网络连接 | 必须 | 必须 | 首次启动需要访问 GitHub；LSP 工具会通过 Mason 下载。 |

如果缺少 Git，Neovim 仍可启动，但插件管理器无法自动安装，很多功能不会加载。

### 按功能可选安装

以下依赖不是启动 Neovim 的必要条件，但会影响对应功能。

| 功能 | 需要安装 | 说明 |
| --- | --- | --- |
| 运行 C/C++ 当前文件 | `g++` 或 `clang++` | 使用 `空格 rr` / `:RunFile` 时需要。LSP 的 `clangd` 不等于 C++ 编译器。 |
| Linux 系统剪贴板 | `wl-clipboard` 或 `xclip` / `xsel` | Wayland 推荐 `wl-clipboard`；X11 可用 `xclip` 或 `xsel`。没有这些工具时，本配置不会强行启用系统剪贴板。 |

Windows 下系统剪贴板通常不需要额外安装工具。

## 不需要提前安装

以下内容不需要用户手动提前安装：

| 组件 | 安装方式 |
| --- | --- |
| Neovim 插件 | 首次启动时由 `lazy.nvim` 自动下载。 |
| Lua LSP (`lua_ls`) | 由 Mason 自动安装。 |
| C/C++ LSP (`clangd`) | 由 Mason 自动安装。 |
| `.clang-format` | 打开 C/C++ 文件时，如果当前 workspace 没有该文件，配置会自动生成默认版本。 |

默认 `.clang-format` 内容为：

```yaml
BasedOnStyle: Google
Standard: Latest
IndentWidth: 4
ColumnLimit: 120
AccessModifierOffset: -4
InsertBraces: true
```

如果项目已经存在 `.clang-format`，配置不会自动覆盖它。

## 配置目录

根据系统不同，Neovim 配置目录通常为：

| 系统 | 配置目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\nvim` |
| Linux | `~/.config/nvim` |

将本仓库内容放入对应目录后，直接启动 `nvim` 即可。

## 首次启动行为

首次启动 Neovim 时会发生以下操作：

1. 如果本地没有 `lazy.nvim`，配置会使用 Git 从 GitHub 克隆它。
2. `lazy.nvim` 会根据配置下载插件。
3. 打开 Lua 或 C/C++ 文件时，Mason 会准备对应 LSP 工具。
4. 打开 C/C++ 文件时，会自动应用 4 空格缩进；如果当前 workspace 没有 `.clang-format`，会自动生成默认配置。

首次安装完成后，建议重启一次 Neovim，让插件和 LSP 状态完全刷新。

## 常用验证命令

在 Neovim 中可以使用以下命令检查环境：

```vim
:checkhealth
:checkhealth vim.lsp
:Mason
```

C/C++ 单文件运行：

```text
空格 rr
```

如果提示找不到编译器，请先安装 `g++` 或 `clang++`，并确保它们在系统 `PATH` 中。
