#!/bin/bash

echo "# stock" >> README.md
git init
git add README.md
git commit -m "datablood commit"
git remote add origin git@github.com:datablood/stock.git
git push -u origin master
