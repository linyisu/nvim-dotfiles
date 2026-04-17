return {
  {
    "xeluxee/competitest.nvim",
    dependencies = { "MunifTanjim/nui.nvim" },
    cmd = "CompetiTest",
    init = function()
      vim.api.nvim_create_autocmd("BufWinEnter", {
        callback = function(ev)
          local ok = pcall(vim.api.nvim_buf_get_var, ev.buf, "competitest_title")
          if ok then
            vim.wo.numberwidth = 1
            vim.wo.signcolumn = "no"
          end
        end,
      })
      vim.api.nvim_create_autocmd("WinResized", {
        callback = function()
          -- find si and so windows by competitest_title buffer var
          local si_win, so_win
          for _, winid in ipairs(vim.api.nvim_list_wins()) do
            local ok, title = pcall(vim.api.nvim_buf_get_var, vim.api.nvim_win_get_buf(winid), "competitest_title")
            if ok then
              if title == "Input" then si_win = winid end
              if title == "Output" then so_win = winid end
            end
          end
          if si_win and so_win then
            local total = vim.api.nvim_win_get_width(si_win) + vim.api.nvim_win_get_width(so_win)
            local half = math.floor(total / 2)
            vim.api.nvim_win_set_width(si_win, half)
          end
        end,
      })
    end,
    opts = {
      testcases_use_single_file = true,
      runner_ui = {
        interface = "split",
      },
      split_ui = {
        position = "right",
        relative_to_editor = true,
        total_width = 0.3,
        vertical_layout = {
          { 1, "tc" },
          { 3, {
            { 1, {
              { 1, "si" },
              { 1, "se" },
            }},
            { 1, {
              { 1, "so" },
              { 1, "eo" },
            }},
          }},
        },
      },
    },
  },
}
