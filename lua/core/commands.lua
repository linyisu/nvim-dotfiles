vim.api.nvim_create_user_command("ClangFormatInit", function(command)
  require("core.clang_format").write_default({ force = command.bang })
end, {
  bang = true,
  desc = "Create default .clang-format in the current workspace",
})

vim.api.nvim_create_user_command("ClangFormatOpen", function()
  require("core.clang_format").open()
end, {
  desc = "Open .clang-format in the current workspace",
})

vim.api.nvim_create_user_command("RunFile", function()
  require("core.run").current_file()
end, {
  desc = "Compile and run the current file",
})

vim.api.nvim_create_user_command("Home", function()
  require("core.home").open()
end, {
  desc = "Open the start page",
})
