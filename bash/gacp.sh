#!/bin/bash
# Git Add Commit Push
#
# git add <file-name>
# git commit -m <"commit message">
# git push origin <bracnh-name|default=master>
#
git add $1
git commit -m $2
git push origin master
