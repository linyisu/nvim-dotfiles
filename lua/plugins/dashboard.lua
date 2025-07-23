return {
    {
        'nvimdev/dashboard-nvim',
        event = 'VimEnter',
        config = function()
            require('dashboard').setup({
                config = {
                    center = { 'header' },
                    header = {
                        '',
                        '',
                        '██╗      ██╗ ███╗   ██╗ ██╗   ██╗ ██╗ ███████╗ ██╗   ██╗',
                        '██║      ██║ ████╗  ██║ ╚██╗ ██╔╝ ██║ ██╔════╝ ██║   ██║',
                        '██║      ██║ ██╔██╗ ██║  ╚████╔╝  ██║ ███████╗ ██║   ██║',
                        '██║      ██║ ██║╚██╗██║   ╚██╔╝   ██║ ╚════██║ ██║   ██║',
                        '███████╗ ██║ ██║ ╚████║    ██║    ██║ ███████║ ╚██████╔╝',
                        '╚══════╝ ╚═╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝  ╚═════╝ ',
                    }
                }
            })
        end,
        dependencies = { { 'nvim-tree/nvim-web-devicons' } }
    }
}
