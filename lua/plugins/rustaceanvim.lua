return {
  {
    'mrcjkb/rustaceanvim',
    version = '^5',
    lazy = false,
    opts = function(_, opts)
      local astrolsp_avail, astrolsp = pcall(require, "astrolsp")
      local astrolsp_opts = (astrolsp_avail and astrolsp.lsp_opts "rust_analyzer") or {}
      local server = {
        settings = function(project_root, default_settings)
          local astrolsp_settings = astrolsp_opts.settings or {}

          local merge_table = require("astrocore").extend_tbl(default_settings or {}, astrolsp_settings)
          local ra = require "rustaceanvim.config.server"
          return ra.load_rust_analyzer_settings(project_root, {
            settings_file_pattern = "rust-analyzer.json",
            default_settings = merge_table,
          })
        end,
      }
      return { server = require("astrocore").extend_tbl(astrolsp_opts, server) }
    end,
    config = function(_, opts)
      if vim.fn.has "nvim-0.12" == 1 then
        local compat_ok, compat = pcall(require, "rustaceanvim.compat")
        if compat_ok then
          compat.client_request = function(client, method, params, handler, bufnr)
            return client:request(method, params, handler, bufnr)
          end
          compat.client_notify = function(client, method, params)
            return client:notify(method, params)
          end
        end
      end
      vim.g.rustaceanvim = require("astrocore").extend_tbl(opts, vim.g.rustaceanvim)
    end,
  },
  {
    "AstroNvim/astrolsp",
    opts = {
      handlers = { rust_analyzer = false },
    },
  },
  {
    "WhoIsSethDaniel/mason-tool-installer.nvim",
    opts = {
      ensure_installed = { "rust-analyzer" },
    },
  },
}
