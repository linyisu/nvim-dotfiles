local is_windows = package.config:sub(1, 1) == "\\"
local uv = vim.uv or vim.loop
local home = vim.fs.normalize((uv.os_homedir and uv.os_homedir()) or vim.fn.expand("~"))
local acm_dir = vim.fs.joinpath(home, "acm")
local cpp_binary = is_windows and "$(FNOEXT).exe" or "$(FNOEXT)"
local cpp_runner = is_windows and "./$(FNOEXT).exe" or "./$(FNOEXT)"

local spec = { "xeluxee/competitest.nvim", name = "competitest.nvim" }

spec.dependencies = { "MunifTanjim/nui.nvim" }
spec.cmd = "CompetiTest"
spec.keys = {
  { "<leader>tt", "<cmd>CompetiTest run<cr>", desc = "Run testcases" },
  { "<leader>tT", "<cmd>CompetiTest run_no_compile<cr>", desc = "Run testcases without compile" },
  { "<leader>ta", "<cmd>CompetiTest add_testcase<cr>", desc = "Add testcase" },
  { "<leader>te", "<cmd>CompetiTest edit_testcase<cr>", desc = "Edit testcase" },
  { "<leader>td", "<cmd>CompetiTest delete_testcase<cr>", desc = "Delete testcase" },
  { "<leader>ts", "<cmd>CompetiTest show_ui<cr>", desc = "Show testcase UI" },
  { "<leader>tr", "<cmd>CompetiTest receive testcases<cr>", desc = "Receive testcases" },
  { "<leader>tp", "<cmd>CompetiTest receive problem<cr>", desc = "Receive problem" },
  { "<leader>tc", "<cmd>CompetiTest receive contest<cr>", desc = "Receive contest" },
}

spec.init = function()
  local group = vim.api.nvim_create_augroup("CompetiTestLayout", { clear = true })

  vim.api.nvim_create_autocmd("WinResized", {
    group = group,
    callback = function()
      local wins = {}

      for _, winid in ipairs(vim.api.nvim_list_wins()) do
        local ok, title = pcall(vim.api.nvim_buf_get_var, vim.api.nvim_win_get_buf(winid), "competitest_title")

        if ok then
          wins[title] = winid
        end
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
    group = group,
    callback = function(ev)
      vim.schedule(function()
        local ok, title = pcall(vim.api.nvim_buf_get_var, ev.buf, "competitest_title")

        if not ok then
          return
        end

        local winid = vim.fn.bufwinid(ev.buf)

        if winid ~= -1 then
          vim.wo[winid].winbar = "%#TabLineFill#%=%#TabLineSel# " .. title .. " %#TabLineFill#%="
        end
      end)
    end,
  })
end

spec.opts = {
  testcases_use_single_file = true,
  received_files_extension = "cpp",
  received_contests_prompt_extension = false,
  compile_command = {
    cpp = {
      exec = "g++",
      args = { "-std=c++23", "-O2", "-Wall", "$(FNAME)", "-o", cpp_binary },
    },
  },
  run_command = {
    cpp = { exec = cpp_runner },
  },
  received_problems_path = vim.fs.joinpath(acm_dir, "problems", "$(JUDGE)", "$(PROBLEM)", "$(PROBLEM).$(FEXT)"),
  received_contests_directory = vim.fs.joinpath(acm_dir, "contests", "$(JUDGE)", "$(CONTEST)"),
  received_contests_problems_path = "$(PROBLEM)/$(PROBLEM).$(FEXT)",
  runner_ui = {
    interface = "split",
  },
  split_ui = {
    position = "right",
    relative_to_editor = true,
    total_width = 0.4,
    vertical_layout = {
      {
        1,
        {
          {
            1,
            {
              { 1, "tc" },
              { 1, "si" },
            },
          },
          { 1, "so" },
        },
      },
      {
        1,
        {
          { 1, "se" },
          { 1, "eo" },
        },
      },
    },
  },
}

spec.config = function(_, opts)
  require("competitest").setup(opts)
end

return {
  spec,
}
