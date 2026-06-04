return {
  {
    "nvim-treesitter/nvim-treesitter",
    opts = {
      ensure_installed = {
        "cpp",
        "rust",
        "toml",
      },
    },
  },

  {
    "neovim/nvim-lspconfig",
    opts = function(_, opts)
      opts.servers = opts.servers or {}

      local lua_ls = opts.servers.lua_ls == true and {} or opts.servers.lua_ls or {}
      opts.servers.lua_ls = vim.tbl_deep_extend("force", lua_ls, {
        settings = {
          Lua = {
            completion = {
              callSnippet = "Replace",
            },
            diagnostics = {
              globals = { "vim", "LazyVim", "Snacks" },
            },
            workspace = {
              checkThirdParty = false,
            },
          },
        },
      })

      local clangd = opts.servers.clangd == true and {} or opts.servers.clangd or {}
      opts.servers.clangd = vim.tbl_deep_extend("force", clangd, {
        cmd = {
          "clangd",
          "--background-index",
          "--clang-tidy",
          "--completion-style=detailed",
          "--fallback-style=Google",
          "--header-insertion=never",
        },
        filetypes = { "c", "cpp", "objc", "objcpp", "cuda" },
      })

      local rust_analyzer = opts.servers.rust_analyzer == true and {} or opts.servers.rust_analyzer or {}
      opts.servers.rust_analyzer = vim.tbl_deep_extend("force", rust_analyzer, {
        settings = {
          ["rust-analyzer"] = {
            cargo = {
              allFeatures = true,
            },
            diagnostics = {
              enable = true,
            },
            procMacro = {
              enable = true,
            },
          },
        },
      })

      return opts
    end,
  },
}
