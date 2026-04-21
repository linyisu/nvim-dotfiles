return {
  {
    "xeluxee/competitest.nvim",
    -- dir = "C:/Users/linyi/Projects/Lua/competitest.nvim",
    dependencies = { "MunifTanjim/nui.nvim" },
    cmd = "CompetiTest",
    -- ft = { "cpp" },
    -- wrap_windows = { "se" },
    init = function()
      vim.api.nvim_create_autocmd("WinResized", {
        callback = function()
          local wins = {}
          for _, winid in ipairs(vim.api.nvim_list_wins()) do
            local ok, title = pcall(vim.api.nvim_buf_get_var, vim.api.nvim_win_get_buf(winid), "competitest_title")
            if ok then wins[title] = winid end
          end
          local tc, so = wins["Testcases"], wins["Output"]
          local se, eo = wins["Errors"], wins["Expected Output"]
          if tc and so then
            local half = math.floor((vim.api.nvim_win_get_width(tc) + vim.api.nvim_win_get_width(so)) / 2)
            vim.api.nvim_win_set_width(tc, half)
          end
          if se and eo then
            local half = math.floor((vim.api.nvim_win_get_width(se) + vim.api.nvim_win_get_width(eo)) / 2)
            vim.api.nvim_win_set_width(se, half)
          end
        end,
      })
      vim.api.nvim_create_autocmd("BufWinEnter", {
        callback = function(ev)
          local buf = ev.buf
          vim.schedule(function()
            local ok, title = pcall(vim.api.nvim_buf_get_var, buf, "competitest_title")
            if ok then
              local winid = vim.fn.bufwinid(buf)
              if winid ~= -1 then
                vim.wo[winid].winbar = "%#TabLineFill#%=%#TabLineSel# " .. title .. " %#TabLineFill#%="
              end
            end
          end)
        end,
      })
    end,

    opts = {
      testcases_use_single_file = true,
      received_files_extension = "cpp",
      received_contests_prompt_extension = false,
      compile_command = {
        cpp = { exec = "g++", args = { "-std=c++23", "-O2", "-Wall", "$(FNAME)", "-o", "$(FNOEXT)" } },
      },
      received_problems_path = "$(HOME)\\acm\\problems\\$(JUDGE)\\$(PROBLEM)\\$(PROBLEM).$(FEXT)",
      received_contests_directory = "$(HOME)\\acm\\contests\\$(JUDGE)\\$(CONTEST)",
      received_contests_problems_path = "$(PROBLEM)\\$(PROBLEM).$(FEXT)",
      runner_ui = {
        interface = "split",
      },
      split_ui = {
        position = "right",
        relative_to_editor = true,
        total_width = 0.4,
        -- auto_open = true,
        vertical_layout = {
          {
            1,
            {
              { 1, {
                { 1, "tc" },
                { 1, "si" },
              } },
              { 1, "so" },
            },
          },
          { 1, {
            { 1, "se" },
            { 1, "eo" },
          } },
        },
      },
    },
  },
}
