" various settings
set shell=/bin/bash
set nocompatible
set number
set nowrap
set splitbelow
set wildmenu
set backspace=indent,eol,start
set mouse=a

" set tab to 4 spaces
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent

" folding options
set foldmethod=syntax
set foldnestmax=10
set foldlevel=2
set nofoldenable

" syntax highlighting
filetype off
filetype plugin indent on
syntax on
" don't highlight all pattern matches
"set is hls
set is
" import color scheme
source ~/.vim/colors.vim
" python syntax highlighting options
let python_self_cls_highlight = 1

" plugins management using vundle
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'kien/ctrlp.vim'
Plugin 'scrooloose/nerdtree'
Plugin 'jiangmiao/auto-pairs'
Plugin 'preservim/nerdcommenter'
Plugin 'fatih/vim-go'
Plugin 'elzr/vim-json'
Plugin 'rust-lang/rust.vim'
Plugin 'kh3phr3n/python-syntax'
Plugin 'pangloss/vim-javascript'
Plugin 'justinmk/vim-syntax-extra'
Plugin 'iamcco/markdown-preview.nvim'
Plugin 'hashivim/vim-terraform'
Plugin 'pearofducks/ansible-vim'
Plugin 'acarapetis/vim-sh-heredoc-highlighting'
call vundle#end()

" restore cursor position to where it was before
augroup JumpCursorOnEdit
   au!
   autocmd BufReadPost *
            \ if expand('<afile>:p:h') !=? $TEMP |
            \   if line("'\"") > 1 && line("'\"") <= line('$') |
            \     let JumpCursorOnEdit_foo = line("'\"") |
            \     let b:doopenfold = 1 |
            \     if (foldlevel(JumpCursorOnEdit_foo) > foldlevel(JumpCursorOnEdit_foo - 1)) |
            \        let JumpCursorOnEdit_foo = JumpCursorOnEdit_foo - 1 |
            \        let b:doopenfold = 2 |
            \     endif |
            \     exe JumpCursorOnEdit_foo |
            \   endif |
            \ endif
   " need to postpone using 'zv' until after reading the modelines.
   autocmd BufWinEnter *
            \ if exists('b:doopenfold') |
            \   exe 'normal zv' |
            \   if(b:doopenfold > 1) |
            \       exe  '+'.1 |
            \   endif |
            \   unlet b:doopenfold |
            \ endif
augroup END

" transform csv files to a more readable outline
command Csv let width=25 | 
            \ let fill = repeat(' ', width) | 
            \ %s/\([^,]*\),\=/\=strpart(submatch(1).fill, 0, width)/ge | 
            \ %s/\s\+$//ge | 
            \ set cursorline

" put parenthesis around selected text
xnoremap <leader>s xi()<Esc>P

" add python virtualenv support, note that active_this.py will be created 
" when using virtualenv command as apposed to `python -m venv <venv>`
py3 <<EOF
import os
import sys 
if 'VIRTUAL_ENV' in os.environ:
    project_base_dir = os.environ['VIRTUAL_ENV']
    activate_this = os.path.join(project_base_dir, 'bin/activate_this.py')
    with open(activate_this, 'rb') as srcfile:
        code = compile(srcfile.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))
EOF

" open a NERDTree if no files were specified on startup
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 0 && !exists('s:std_in') | NERDTree | endif
" workaround for directory arrows displayed as blocks
let g:NERDTreeDirArrowExpandable = '+'
let g:NERDTreeDirArrowCollapsible = '~'
map <C-q> :NERDTreeToggle<CR>

" ycm options
let g:ycm_show_diagnostics_ui = 0
let g:ycm_autoclose_preview_window_after_completion = 1 
let g:ycm_disable_for_files_larger_than_kb = 10000
let g:ycm_always_populate_location_list = 1
"let g:ycm_python_binary_path = '~/.pyenvs/dpl/bin/python'
let g:ycm_server_python_interpreter = '/usr/bin/python3'
let g:ycm_global_ycm_extra_conf = '~/.vim/ycm_extra_conf.py'
"let g:ycm_server_log_level = 'debug'
"let g:enable_ycm_at_startup = 0
