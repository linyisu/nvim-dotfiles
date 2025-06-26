-- Snippets 管理工具
-- 提供一些实用的命令来管理代码片段

local M = {}
local snippet_manager = require("snippets")

-- 重新加载所有代码片段
function M.reload_snippets()
    -- 清除现有的代码片段
    local ls = require("luasnip")
    ls.cleanup()
    
    -- 静默重新加载所有代码片段
    snippet_manager.load_all_snippets()
end

-- 测试snippet加载
function M.test_snippet_loading()
    print("Testing snippet loading...")
    
    -- 获取当前文件类型
    local filetype = vim.bo.filetype
    if filetype == "" then
        filetype = "cpp"  -- 默认测试cpp
    end
    
    local ls = require("luasnip")
    local snippets = ls.get_snippets(filetype)
    
    if snippets and #snippets > 0 then
        print("Found " .. #snippets .. " snippets for filetype: " .. filetype)
        print("Sample snippets:")
        for i = 1, math.min(5, #snippets) do
            if snippets[i].trigger then
                print("  - " .. snippets[i].trigger)
            end
        end
    else
        print("No snippets found for filetype: " .. filetype)
    end
end

-- 带输出的调试加载函数
function M.reload_snippets_verbose()
    print("正在重新加载所有代码片段...")
    
    -- 清除现有的代码片段
    local ls = require("luasnip")
    ls.cleanup()
    
    -- 临时启用详细模式加载
    local original_load = snippet_manager.load_all_snippets
    snippet_manager.load_all_snippets = function()
        local ls = require("luasnip")
        local snippet_files = snippet_manager.get_snippet_files()
        
        for _, file in ipairs(snippet_files) do
            local success, snippets_or_err = pcall(require, "snippets." .. file)
            if success then
                if type(snippets_or_err) == "table" then
                    local lang = snippet_manager.get_language_from_filename(file)
                    
                    if string.find(file, "generated_snippets") then
                        print("加载 " .. file .. " (自动检测语言)")
                        ls.add_snippets("cpp", snippets_or_err) -- 简化为直接添加到cpp
                    else
                        ls.add_snippets(lang, snippets_or_err)
                        print("加载 " .. file .. " 到语言: " .. lang)
                    end
                end
            else
                print("错误加载文件: " .. file .. " - " .. tostring(snippets_or_err))
            end
        end
    end
    
    -- 加载snippets
    snippet_manager.load_all_snippets()
    
    -- 恢复原始函数
    snippet_manager.load_all_snippets = original_load
    
    print("所有代码片段已重新加载完成")
end

-- 列出所有已加载的 snippet 文件
function M.list_snippet_files()
    local files = snippet_manager.get_snippet_files()
    print("已加载的 snippet 文件:")
    for i, file in ipairs(files) do
        print(i .. ". " .. file)
    end
end

-- 添加新的 snippet 文件
function M.add_snippet_file(filename)
    snippet_manager.add_snippet_file(filename)
end

-- 移除 snippet 文件
function M.remove_snippet_file(filename)
    snippet_manager.remove_snippet_file(filename)
end

-- 创建用户命令
vim.api.nvim_create_user_command('SnippetsReload', M.reload_snippets, {})
vim.api.nvim_create_user_command('SnippetsReloadVerbose', M.reload_snippets_verbose, {})
vim.api.nvim_create_user_command('SnippetsList', M.list_snippet_files, {})
vim.api.nvim_create_user_command('SnippetsTest', M.test_snippet_loading, {})
vim.api.nvim_create_user_command('SnippetsAdd', function(opts)
    M.add_snippet_file(opts.args)
end, {nargs = 1})
vim.api.nvim_create_user_command('SnippetsRemove', function(opts)
    M.remove_snippet_file(opts.args)
end, {nargs = 1})

return M
