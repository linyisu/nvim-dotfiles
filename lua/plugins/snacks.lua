return {
  {
    "folke/snacks.nvim",
    opts = {
      dashboard = {
        preset = {
          keys = {
            { key = "f", action = ":lua Snacks.dashboard.pick('files')", icon = "", desc = "Find File" },
            { key = "r", action = ":lua Snacks.dashboard.pick('oldfiles')", icon = "", desc = "Recent Files" },
            { key = "p", action = ":lua Snacks.dashboard.pick('projects')", icon = "", desc = "Projects" },
            { key = "e", action = "<Leader>e", icon = "󰙅", desc = "Explorer" },
            {
              key = "c",
              action = ":lua Snacks.dashboard.pick('files', {cwd = vim.fn.stdpath('config')})",
              icon = "",
              desc = "Config",
            },
            { key = "s", section = "session", icon = "", desc = "Restore Session" },
            { key = "l", action = ":Lazy", enabled = package.loaded.lazy ~= nil, icon = "󰒲", desc = "Lazy" },
            { key = "q", action = ":qa", icon = "", desc = "Quit" },
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
