return {
    {
        "L3MON4D3/LuaSnip",
        version = "v2.*",                -- 使用最新的 v2 版本
        build = "make install_jsregexp", -- 安装 jsregexp（可选）

        -- 插件加载完成后，执行以下配置
        config = function()
            -- 这里的路径需要根据实际情况调整，假设 snippets 文件夹在同级
            -- require("snippets.cpp")  -- 加载 C++ 语言片段
            -- require("snippets.fast")  -- 加载便携代码片段
            require("snippets.generated_snippets1")  -- 加载所有语言片段
            require("snippets.generated_snippets2")  -- 加载片段
            require("snippets.generated_snippets3")  -- 加载片段
            require("snippets.generated_snippets4")  -- 加载片段
            require("snippets.generated_snippets5")  -- 加载片段
            require("snippets.generated_snippets6")  -- 加载片段
            require("snippets.generated_snippets7")  -- 加载片段
            require("snippets.generated_snippets8")  -- 加载片段
            require("snippets.generated_snippets9")  -- 加载片段
            require("snippets.generated_snippets10")  -- 加载片段
        end
    }
}

