return {
  {
    "garymjr/nvim-snippets",
    optional = true,
    enabled = false,
  },

  {
    "rafamadriz/friendly-snippets",
    optional = true,
    enabled = false,
  },

  {
    "L3MON4D3/LuaSnip",
    event = "InsertEnter",
    opts = {
      history = true,
      delete_check_events = "TextChanged",
    },
    config = function(_, opts)
      local ls = require("luasnip")

      ls.setup(opts)

      LazyVim.cmp.actions.snippet_forward = function()
        if ls.jumpable(1) then
          vim.schedule(function()
            ls.jump(1)
          end)
          return true
        end
      end

      LazyVim.cmp.actions.snippet_stop = function()
        if ls.expand_or_jumpable() then
          ls.unlink_current()
          return true
        end
      end

      require("core.cpp_snippets").setup()
    end,
  },

  {
    "saghen/blink.cmp",
    optional = true,
    opts = {
      keymap = {
        preset = "super-tab",
      },
      snippets = {
        preset = "luasnip",
      },
    },
  },
}
