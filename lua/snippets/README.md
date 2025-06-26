# Snippets 管理系统

这个目录包含了一个统一的代码片段管理系统，用于 LuaSnip 插件。

## 文件结构

```
snippets/
├── init.lua          # 主要的 snippets 管理器
├── manager.lua       # 管理工具和用户命令
├── cpp.lua          # C++ 代码片段
├── fast.lua         # 快速代码片段
└── generated_snippets*.lua  # 生成的代码片段文件
```

## 主要功能

### 1. 智能加载 snippets
系统支持两种snippet文件格式：
- **直接注册格式**: 文件中直接调用 `ls.add_snippets()`
- **返回数组格式**: 文件返回snippet数组，系统自动检测语言并注册

### 2. 自动语言检测
- 根据文件名自动推断语言类型 (如 `cpp.lua` → `cpp`)
- 对于 `generated_snippets` 文件，会分析snippet内容来检测语言
- 支持将snippet分配到正确的语言环境

### 3. 管理命令
提供了以下 Neovim 命令来管理代码片段：

- `:SnippetsReload` - 重新加载所有代码片段
- `:SnippetsList` - 列出所有已加载的 snippet 文件
- `:SnippetsTest` - 测试当前文件类型的snippet加载情况
- `:SnippetsAdd <filename>` - 添加新的 snippet 文件到加载列表
- `:SnippetsRemove <filename>` - 从加载列表中移除 snippet 文件

### 3. 快捷键
以下快捷键已在 LuaSnip 配置中设置：

- `Ctrl+K` (插入模式) - 展开代码片段
- `Ctrl+L` (插入/选择模式) - 跳转到下一个占位符
- `Ctrl+J` (插入/选择模式) - 跳转到上一个占位符
- `Ctrl+E` (插入/选择模式) - 在选择项中切换

## 使用方法

### 添加新的代码片段文件

1. 在 `snippets/` 目录下创建新的 `.lua` 文件
2. 使用命令 `:SnippetsAdd <filename>` 将其添加到加载列表
3. 使用 `:SnippetsReload` 重新加载所有代码片段

### 修改现有代码片段

1. 编辑对应的 `.lua` 文件
2. 使用 `:SnippetsReload` 重新加载代码片段

### 示例：创建新的代码片段文件

```lua
-- 文件: snippets/python.lua
local ls = require("luasnip")
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node

ls.add_snippets("python", {
    s("def", {
        t("def "), i(1, "function_name"), t("("), i(2), t("):"),
        t({"", "    "}), i(0)
    }),
    s("class", {
        t("class "), i(1, "ClassName"), t(":"),
        t({"", "    def __init__(self"}), i(2), t("):"),
        t({"", "        "}), i(0)
    })
})
```

然后使用：
```
:SnippetsAdd python
:SnippetsReload
```

### 示例：测试当前配置

完成配置后，你可以：

1. 重启 Neovim 或运行 `:SnippetsReload`
2. 打开一个 `.cpp` 文件
3. 运行 `:SnippetsTest` 查看加载的snippet
4. 尝试输入 `vii` 然后按 `Ctrl+K` 来测试snippet展开

### 支持的文件格式

#### 返回数组格式 (如 fast.lua)
```lua
local ls = require("luasnip")
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node

return {
    s("trigger", { t("content"), i(1) }),
    -- 更多snippets...
}
```

#### 直接注册格式 (如 cpp.lua)
```lua
local ls = require("luasnip")
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node

ls.add_snippets("cpp", {
    s("trigger", { t("content"), i(1) }),
    -- 更多snippets...
})
```

## 注意事项

- 代码片段文件应该遵循 LuaSnip 的语法规范
- 文件名不需要包含 `.lua` 扩展名
- 系统会自动处理加载错误并显示错误信息
- 所有更改会在下次重启 Neovim 时自动加载
