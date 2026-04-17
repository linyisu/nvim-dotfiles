-- if true then return {} end -- WARN: REMOVE THIS LINE TO ACTIVATE THIS FILE

---@type LazySpec
return {

  "andweeb/presence.nvim",
  {
    "echasnovski/mini.move",
    opts = {
      mappings = {
        up         = "<A-k>",
        down       = "<A-j>",
        line_up    = "<A-k>",
        line_down  = "<A-j>",
        left       = "",
        right      = "",
        line_left  = "",
        line_right = "",
      },
    },
  },
  {
    "nvim-treesitter/nvim-treesitter",
    opts = {
      ensure_installed = { "html" },
    },
  },
  {
    "ray-x/lsp_signature.nvim",
    event = "BufRead",
    config = function() require("lsp_signature").setup() end,
  },

  {
    "folke/snacks.nvim",
    opts = {
      dashboard = {
        preset = {
          keys = {
            { key = "f", action = "<Leader>ff",  icon = "", desc = "Find File"       },
            { key = "p", action = "<Leader>cpr", icon = "", desc = "Receive Problem" },
            { key = "C", action = "<Leader>cpc", icon = "", desc = "Receive Contest" },
            { key = "e", action = "<Leader>ta",  icon = "", desc = "Leetcode"        },
            { key = "m", action = "<Leader>m",   icon = "", desc = "Zoxide"          },
            { key = "c", icon = "", desc = "Config", action = ":lua Snacks.dashboard.pick('files', {cwd = vim.fn.stdpath('config')})" },
            { key = "l", icon = "󰒲", desc = "Lazy", action = ":Lazy", enabled = package.loaded.lazy ~= nil },
          },
          header = table.concat({
            "██╗     ██╗███╗   ██╗██╗   ██╗██╗███████╗██╗   ██╗",
            "██║     ██║████╗  ██║╚██╗ ██╔╝██║██╔════╝██║   ██║",
            "██║     ██║██╔██╗ ██║ ╚████╔╝ ██║███████╗██║   ██║",
            "██║     ██║██║╚██╗██║  ╚██╔╝  ██║╚════██║██║   ██║",
            "███████╗██║██║ ╚████║   ██║   ██║███████║╚██████╔╝",
            "╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝ ╚═════╝ ",
          }, "\n"),
        },
      },
    },
  },
}
