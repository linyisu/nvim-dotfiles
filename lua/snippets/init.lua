-- Snippets 统一管理文件
-- 这个文件负责加载所有的代码片段

local M = {}

-- 定义所有的 snippet 文件
local snippet_files = {
    "cpp",                    -- C++ 代码片段
    "fast",                   -- 快速代码片段
    "generated_snippets1",    -- 生成的代码片段 1
    "generated_snippets2",    -- 生成的代码片段 2
    "generated_snippets3",    -- 生成的代码片段 3
    "generated_snippets4",    -- 生成的代码片段 4
    "generated_snippets5",    -- 生成的代码片段 5
    "generated_snippets6",    -- 生成的代码片段 6
    "generated_snippets7",    -- 生成的代码片段 7
    "generated_snippets8",    -- 生成的代码片段 8
    "generated_snippets9",    -- 生成的代码片段 9
    "generated_snippets10",   -- 生成的代码片段 10
}

-- 加载所有代码片段的函数
function M.load_all_snippets()
    local ls = require("luasnip")
    
    for _, file in ipairs(snippet_files) do
        local success, snippets_or_err = pcall(require, "snippets." .. file)
        if success then
            -- 检查返回值类型
            if type(snippets_or_err) == "table" then
                -- 如果文件返回了一个snippet数组，我们需要将它们添加到相应的语言
                local lang = M.get_language_from_filename(file)
                
                -- 对于generated_snippets文件，可能包含多种语言的片段
                if string.find(file, "generated_snippets") then
                    -- 静默检测snippet中的语言并分别添加
                    M.add_snippets_by_detection(snippets_or_err)
                else
                    -- 普通文件直接添加到推断的语言
                    ls.add_snippets(lang, snippets_or_err)
                end
            end
            -- 移除了所有print语句，实现静默加载
        end
        -- 移除了错误输出，静默处理错误
    end
end

-- 通过检测snippet内容来添加到对应语言
function M.add_snippets_by_detection(snippets)
    local ls = require("luasnip")
    
    -- 将所有snippet按语言分组
    local lang_groups = {}
    
    for _, snippet in ipairs(snippets) do
        -- 尝试从snippet的trigger名称或内容检测语言
        local lang = M.detect_snippet_language(snippet)
        
        if not lang_groups[lang] then
            lang_groups[lang] = {}
        end
        table.insert(lang_groups[lang], snippet)
    end
    
    -- 为每种语言添加对应的snippets，静默处理
    for lang, lang_snippets in pairs(lang_groups) do
        ls.add_snippets(lang, lang_snippets)
    end
end

-- 检测单个snippet的语言
function M.detect_snippet_language(snippet)
    -- 获取snippet的trigger名称
    local trigger = ""
    if snippet and snippet.trigger then
        trigger = snippet.trigger
    end
    
    -- 检查snippet内容中的文本节点来检测语言
    local content = ""
    if snippet and snippet.nodes then
        for _, node in ipairs(snippet.nodes) do
            if node.text then
                if type(node.text) == "table" then
                    content = content .. table.concat(node.text, " ")
                elseif type(node.text) == "string" then
                    content = content .. node.text
                end
            end
        end
    end
    
    -- 根据trigger和内容特征检测语言
    local combined_text = (trigger .. " " .. content):lower()
    
    -- C++ 特征检测
    if string.find(combined_text, "#include") or
       string.find(combined_text, "using namespace") or
       string.find(combined_text, "vector<") or
       string.find(combined_text, "std::") or
       string.find(combined_text, "class ") or
       string.find(combined_text, "public:") or
       string.find(combined_text, "private:") or
       string.find(combined_text, "int main") or
       string.find(combined_text, "#pragma") or
       string.find(trigger:lower(), "cpp") or 
       string.find(trigger:lower(), "c++") or
       string.find(trigger:lower(), "_h$") or
       string.find(trigger:lower(), "_cpp$") then
        return "cpp"
    
    -- Python 特征检测
    elseif string.find(combined_text, "def ") or
           string.find(combined_text, "import ") or
           string.find(combined_text, "from ") or
           string.find(combined_text, "class ") or
           string.find(trigger:lower(), "py") or 
           string.find(trigger:lower(), "python") then
        return "python"
    
    -- JavaScript/TypeScript 特征检测
    elseif string.find(combined_text, "function") or
           string.find(combined_text, "const ") or
           string.find(combined_text, "let ") or
           string.find(combined_text, "var ") or
           string.find(combined_text, "=>") or
           string.find(trigger:lower(), "js") or 
           string.find(trigger:lower(), "javascript") or
           string.find(trigger:lower(), "ts") or
           string.find(trigger:lower(), "typescript") then
        return "javascript"
    
    -- Java 特征检测
    elseif string.find(combined_text, "public class") or
           string.find(combined_text, "public static void main") or
           string.find(trigger:lower(), "java") then
        return "java"
    
    -- Rust 特征检测
    elseif string.find(combined_text, "fn ") or
           string.find(combined_text, "let mut") or
           string.find(trigger:lower(), "rust") then
        return "rust"
    
    -- Go 特征检测
    elseif string.find(combined_text, "func ") or
           string.find(combined_text, "package ") or
           string.find(trigger:lower(), "go") then
        return "go"
    
    else
        -- 默认为cpp，因为大部分generated_snippets看起来是C++代码
        return "cpp"
    end
end

-- 根据文件名推断语言类型
function M.get_language_from_filename(filename)
    -- 文件名到语言的映射
    local lang_mapping = {
        cpp = "cpp",
        c = "c", 
        python = "python",
        py = "python",
        javascript = "javascript",
        js = "javascript",
        typescript = "typescript",
        ts = "typescript",
        java = "java",
        rust = "rust",
        go = "go",
        lua = "lua",
        fast = "cpp",  -- fast.lua 包含C++代码片段
    }
    
    -- 检查是否有直接匹配
    if lang_mapping[filename] then
        return lang_mapping[filename]
    end
    
    -- 检查是否包含语言名称
    for pattern, lang in pairs(lang_mapping) do
        if string.find(filename:lower(), pattern) then
            return lang
        end
    end
    
    -- 默认返回 "all"，表示适用于所有语言
    return "all"
end

-- 获取所有 snippet 文件列表
function M.get_snippet_files()
    return snippet_files
end

-- 添加新的 snippet 文件
function M.add_snippet_file(filename)
    table.insert(snippet_files, filename)
end

-- 移除 snippet 文件
function M.remove_snippet_file(filename)
    for i, file in ipairs(snippet_files) do
        if file == filename then
            table.remove(snippet_files, i)
            break
        end
    end
end

return M
