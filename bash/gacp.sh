#!/bin/bash
# Git Add Commit Push
#
# git add <file-name>
# git commit -m <"commit message">
# git push origin <bracnh-name|default=master>
#
if ["$1" == ""] || ["$2" == ""]
then
	echo 'Usage: gacp <file-name> "<commit-message>" <branch-name|default=master>'
	exit
fi

git add $1
git commit -m $2
if["$3" == ""]
then
	git push origin master
else
	git push origin $3
fi
