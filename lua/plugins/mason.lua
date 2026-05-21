local function tool_name(tool) return type(tool) == "table" and tool[1] or tool end

local function normalize_tools(tools)
  local ret, seen = {}, {}
  for _, tool in ipairs(tools or {}) do
    local name = tool_name(tool)
    if name and not seen[name] then
      seen[name] = true
      ret[#ret + 1] = tool
    end
  end
  return ret
end

local function add_tool(tools, tool)
  local name = tool_name(tool)
  for _, existing in ipairs(tools) do
    if tool_name(existing) == name then return end
  end
  tools[#tools + 1] = tool
end

return {
  {
    "williamboman/mason.nvim",
    lazy = false,
  },
  {
    "WhoIsSethDaniel/mason-tool-installer.nvim",
    lazy = false,
    opts = function(_, opts)
      opts.ensure_installed = normalize_tools(opts.ensure_installed)

      for _, tool in ipairs {
        "lua-language-server",
        "json-lsp",
        "black",
        "isort",
        "ruff",
        "stylua",
        { "tree-sitter-cli", condition = function() return vim.fn.executable "tree-sitter" == 0 end },
      } do
        add_tool(opts.ensure_installed, tool)
      end

      opts.auto_update = false
      opts.run_on_start = true
      opts.start_delay = 3000
      opts.debounce_hours = nil
      return opts
    end,
  },
}
