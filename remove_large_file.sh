#!/bin/bash

# Script to remove large files from git history using filter-branch

echo "This script will remove Miniconda3-latest-MacOSX-x86_64.sh from git history"
echo "Warning: This will rewrite git history."
echo ""
echo "Press Enter to continue or Ctrl+C to abort..."
read

# Remove the file from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch Miniconda3-latest-MacOSX-x86_64.sh" \
  --prune-empty --tag-name-filter cat -- --all

# Force garbage collection
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Large file has been removed from git history."
echo "You can now push with: git push -f origin main"