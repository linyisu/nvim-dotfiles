return {
  {
    "folke/snacks.nvim",
    opts = {
      dashboard = {
        preset = {
          keys = {
            { key = "p", action = "<Leader>cpr", icon = "", desc = "Receive Problem" },
            { key = "C", action = "<Leader>cpc", icon = "", desc = "Receive Contest" },
            { key = "z", action = "<Leader>m", icon = "", desc = "Zoxide" },
            {
              key = "c",
              action = ":lua Snacks.dashboard.pick('files', {cwd = vim.fn.stdpath('config')})",
              icon = "",
              desc = "Config",
            },
            { key = "l", action = ":Lazy", enabled = package.loaded.lazy ~= nil, icon = "󰒲", desc = "Lazy" },
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
      notifier = {
        width = { min = 40, max = 0.6 },
        height = { min = 1, max = 0.8 },
      },
      image = {
        enabled = false,
        doc = {
          enabled = false,
          inline = false,
          float = false,
        },
      },
      styles = {
        notification = {
          wo = { wrap = true },
        },
      },
    },
    config = function(_, opts)
      local Snacks = require "snacks"
      Snacks.setup(opts)

      for _, name in ipairs { "bigfile", "explorer", "image", "quickfile", "scroll", "statuscolumn", "words" } do
        local ok, mod = pcall(require, "snacks." .. name)
        if ok and mod.meta then mod.meta.health = false end
      end

      local ok, dashboard = pcall(require, "snacks.dashboard")
      if ok and dashboard.health then
        local health = dashboard.health
        dashboard.health = function()
          local warn = Snacks.health.warn
          Snacks.health.warn = function(msg, ...)
            if type(msg) == "string" and msg:match "^dashboard did not open:" then
              return Snacks.health.ok "dashboard skipped outside the start screen"
            end
            return warn(msg, ...)
          end

          local success, err = pcall(health)
          Snacks.health.warn = warn
          if not success then error(err) end
        end
      end
    end,
  },
}
