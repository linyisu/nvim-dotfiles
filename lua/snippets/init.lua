local snippets = {}

local modules = {
    "snippets.cpp",
    "snippets.fast",
    "snippets.generated_snippets1",
    "snippets.generated_snippets2",
    "snippets.generated_snippets3",
    "snippets.generated_snippets4",
    "snippets.generated_snippets5",
    "snippets.generated_snippets6",
    "snippets.generated_snippets7",
    "snippets.generated_snippets8",
    "snippets.generated_snippets9",
    "snippets.generated_snippets10",
}

for _, mod in ipairs(modules) do
    local ok, loaded = pcall(require, mod)
    if ok and type(loaded) == "table" then
        vim.list_extend(snippets, loaded)
    else
        vim.notify("Failed to load snippet module: " .. mod)
    end
end

return snippets
