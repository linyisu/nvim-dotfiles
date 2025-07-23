return {
  'mg979/vim-visual-multi',
  -- 建议在插件加载前通过 init 函数设置好所有配置
  init = function()
    -- 在这里设置所有 vim-visual-multi 的全局变量
    -- 详情可以查看 :help vm-settings

    -- 设置触发多光标模式的主快捷键
    -- 注意: 如果你设置了这个，默认的 <C-n> 可能就不再生效了
    vim.g.vm_leader = '<C-n>' -- 使用 Ctrl+n 作为主快捷键

    -- 自定义多光标模式下的快捷键
    -- 这是一个非常重要的配置，可以让你把功能映射到你习惯的按键上
    vim.g.vm_maps = {
      -- 查找类
      ['Find Under'] = 'n', -- 在当前单词下方查找下一个匹配项并添加光标
      ['Find Next'] = 'n', -- 功能同上
      ['Find Prev'] = 'N', -- 查找上一个匹配项并添加光标
      ['Find All'] = '<C-n>', -- 查找所有匹配项并添加光标 (在多光标模式下)
      ['Skip Region'] = 's', -- 跳过当前匹配项，寻找下一个
      ['Remove Region'] = 'S', -- 移除当前光标/选区

      -- 选择类
      ['Select All'] = 'A', -- 全选

      -- 光标操作
      ['Switch Mode'] = '<Tab>', -- 在 "Visual" 和 "Normal" 模式间切换
      ['Start in Insert'] = 'i', -- 进入插入模式
      ['Add Cursor Down'] = '<C-j>', -- 在下方添加一个光标
      ['Add Cursor Up'] = '<C-k>', -- 在上方添加一个光标
    }

    -- 你可以选择一个预设的主题，或者自定义高亮
    -- 可选主题: 'ocean', 'solarized', 'dracula', 'rigel', 'vscode'
    vim.g.VM_theme = 'dracula'

    -- 如果你不想用预设主题，可以自定义高亮颜色
    -- 下面是自定义高亮的示例 (如果用了 VM_theme, 下面的设置会被覆盖)
    -- vim.g.VM_Mono_hl = { link = 'Comment' }  -- 单光标模式的高亮
    -- vim.g.VM_Cursor_hl = { link = 'Error' }   -- 多光标模式下光标的高亮
    -- vim.g.VM_Region_hl = { link = 'Visual' }  -- 多光标模式下选区的高亮

    -- 允许在哪些文件类型中禁用插件
    vim.g.VM_disabled_filetypes = { 'qf', 'fugitive', 'vista' }
  end,

  -- 如果你不想用 init, 也可以在这里配置快捷键来启动插件
  -- 但官方更推荐用 init 来设置全局变量
  keys = {
    -- 这个快捷键可以在普通模式下启动多光标模式
    {
      '<C-n>',
      '<Plug>(VM-Find-Under)',
      mode = 'n',
      noremap = true,
      silent = true,
      desc = 'VM: Find under cursor',
    },
    -- 这个快捷键可以在可视模式下启动多光标模式
    {
      '<C-n>',
      '<Plug>(VM-Visual-Find)',
      mode = 'x',
      noremap = true,
      silent = true,
      desc = 'VM: Find selected text',
    },
  },
}
