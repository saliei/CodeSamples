" color scheme settings
set background=dark

highlight PreProc  cterm=NONE    ctermfg=cyan      gui=NONE    guifg=cyan
highlight String   cterm=NONE    ctermfg=red       gui=NONE    guifg=red
highlight Comment  cterm=NONE    ctermfg=DarkGray
highlight Type     cterm=NONE    ctermfg=green     gui=NONE    guifg=green
highlight Special  cterm=NONE    ctermfg=Magenta   gui=NONE    guifg=Magenta
highlight SpellBad ctermbg=23    ctermfg=7
highlight Visual   ctermbg=NONE  cterm=reverse     gui=reverse guifg=Grey guibg=fg
highlight LineNr   ctermfg=240
"highlight Normal ctermbg=0
" background color for autocomplete popup
highlight Pmenu    ctermbg=gray  guibg=gray
highlight PmenuSel ctermfg=White guifg=White
" turn off highlight for sign gutter
highlight clear SignColumn
highlight YcmWarningSection ctermbg=Red     ctermfg=black
highlight YcmErrorSection   ctermbg=Red     ctermfg=black
highlight Directory         ctermfg=Blue    guifg=DarkBlue
highlight ErrorMsg          ctermfg=White   guifg=Red
highlight WarningMsg        ctermfg=Red     guifg=Red
highlight MoreMsg           ctermfg=Red     gui=bold        guifg=Red
highlight Title             ctermfg=Blue    gui=bold        guifg=DarkBlue
highlight htmlLink          cterm=Underline ctermfg=Blue    guifg=blue
highlight VertSplit         cterm=NONE      ctermfg=244
"highlight StatusLine       cterm=bold      ctermbg=blue    ctermfg=blue    guibg=blue    guifg=blue

" uncomment next two lines to highlight current line number
"highlight CursorLine cterm=NONE ctermbg=NONE ctermfg=NONE guibg=NONE guifg=NONE
"set cursorline
