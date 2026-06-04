local c_family_filetypes = {
  "c",
  "cpp",
  "cuda",
  "objc",
  "objcpp",
}

local function add_unique(values, value)
  if vim.tbl_contains(values, value) then
    return
  end

  values[#values + 1] = value
end

return {
  {
    "stevearc/conform.nvim",
    opts = function(_, opts)
      opts.formatters_by_ft = opts.formatters_by_ft or {}
      opts.formatters = opts.formatters or {}

      for _, filetype in ipairs(c_family_filetypes) do
        opts.formatters_by_ft[filetype] = { "clang_format" }
      end

      opts.formatters.clang_format = vim.tbl_deep_extend("force", opts.formatters.clang_format or {}, {
        prepend_args = { "--style=file" },
      })

      return opts
    end,
  },

  {
    "mason-org/mason.nvim",
    opts = function(_, opts)
      opts.ensure_installed = opts.ensure_installed or {}
      add_unique(opts.ensure_installed, "clang-format")

      return opts
    end,
  },
}
