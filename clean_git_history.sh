#!/bin/bash

# Script to remove large files from git history

echo "This script will remove Miniconda3-latest-MacOSX-x86_64.sh from git history"
echo "Warning: This will rewrite git history. If you've already pushed to a shared repository,"
echo "you'll need to force push and other users will need to re-clone or rebase."
echo ""
echo "Press Enter to continue or Ctrl+C to abort..."
read

# Create a new branch without the large file
git checkout --orphan temp_branch
git add --all -- ':!Miniconda3-latest-MacOSX-x86_64.sh'
git commit -m "Initial commit without large files"

# Delete the old branch and rename the new one
git branch -D main
git branch -m main

echo ""
echo "Large file has been removed from git history."
echo "You can now push with: git push -f origin main"