git init
git add .gitignore                       # add this FIRST and check it's working
git status                               # CONFIRM secrets.toml is NOT listed
git add .
git status                               # CONFIRM again secrets.toml is absent before committing
git commit -m "Airbnb theme & sentiment recommender - Phase 0-2"