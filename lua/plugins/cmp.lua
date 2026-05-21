return {
  {
    "Saghen/blink.cmp",
    opts = function(_, opts)
      local function has_words_before()
        local line, col = unpack(vim.api.nvim_win_get_cursor(0))
        return col ~= 0 and vim.api.nvim_buf_get_lines(0, line - 1, line, true)[1]:sub(col, col):match "%s" == nil
      end

      opts.keymap = opts.keymap or {}
      opts.keymap["<CR>"] = false
      opts.keymap["<Tab>"] = {
        function(cmp)
          if cmp.is_menu_visible() then
            if cmp.get_selected_item() then return cmp.accept() end
            return cmp.select_next()
          end
        end,
        "snippet_forward",
        function(cmp)
          if has_words_before() or vim.api.nvim_get_mode().mode == "c" then return cmp.show() end
        end,
        "fallback",
      }

      opts.keymap["<S-Tab>"] = {
        "select_prev",
        "snippet_backward",
        function(cmp)
          if vim.api.nvim_get_mode().mode == "c" then return cmp.show() end
        end,
        "fallback",
      }

      return opts
    end,
  },
  { "hrsh7th/nvim-cmp", enabled = false },
}
