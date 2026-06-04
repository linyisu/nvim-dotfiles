local logo = table.concat({
  "██╗     ██╗███╗   ██╗██╗   ██╗██╗███████╗██╗   ██╗",
  "██║     ██║████╗  ██║╚██╗ ██╔╝██║██╔════╝██║   ██║",
  "██║     ██║██╔██╗ ██║ ╚████╔╝ ██║███████╗██║   ██║",
  "██║     ██║██║╚██╗██║  ╚██╔╝  ██║╚════██║██║   ██║",
  "███████╗██║██║ ╚████║   ██║   ██║███████║╚██████╔╝",
  "╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝ ╚═════╝ ",
}, "\n")

return {
  {
    "LazyVim/LazyVim",
    opts = {
      colorscheme = "tokyonight",
    },
  },

  {
    "folke/tokyonight.nvim",
    opts = {
      style = "night",
      terminal_colors = true,
      styles = {
        comments = { italic = true },
        keywords = { italic = false },
      },
    },
  },

  {
    "folke/snacks.nvim",
    opts = function(_, opts)
      opts.scroll = vim.tbl_deep_extend("force", opts.scroll or {}, {
        enabled = false,
      })
      opts.scope = vim.tbl_deep_extend("force", opts.scope or {}, {
        enabled = false,
      })
      opts.indent = opts.indent or {}
      opts.indent.animate = vim.tbl_deep_extend("force", opts.indent.animate or {}, {
        enabled = false,
      })
      opts.indent.scope = vim.tbl_deep_extend("force", opts.indent.scope or {}, {
        enabled = false,
      })

      opts.dashboard = opts.dashboard or {}
      opts.dashboard.preset = opts.dashboard.preset or {}
      opts.dashboard.preset.header = logo
      opts.dashboard.preset.keys = {
        { icon = " ", key = "f", desc = "Find File", action = ":lua require('core.search').pick_files()" },
        { icon = " ", key = "g", desc = "Find Text", action = ":lua require('core.search').pick_grep()" },
        { icon = " ", key = "e", desc = "Explorer", action = ":lua require('core.file_explorer').open_workspace(vim.fn.getcwd())" },
        { icon = " ", key = "n", desc = "New File", action = ":ene | startinsert" },
        { icon = " ", key = "r", desc = "Recent Files", action = ":lua require('core.search').oldfiles()" },
        {
          icon = " ",
          key = "c",
          desc = "Config",
          action = ":lua require('core.file_explorer').open_workspace(vim.fn.stdpath('config'))",
        },
        { icon = "󰒲 ", key = "l", desc = "Lazy", action = ":Lazy" },
        { icon = " ", key = "q", desc = "Quit", action = ":qa" },
      }

      return opts
    end,
  },

  {
    "akinsho/bufferline.nvim",
    opts = function(_, opts)
      opts.options = opts.options or {}
      opts.options.always_show_bufferline = true

      return opts
    end,
  },
}
