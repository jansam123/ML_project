# Getting to [Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660216/Connecting+to+the+system)
Snellius docs: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/74227835/Generic+usage+guides

1. Look for an email from SURFcua with the subject "SURFcua new login scur0034 created". In this email, you will find your username. To get the password, you need to reset [here](https://sso.cua.surf.nl/realms/cua/login-actions/reset-credentials). Follow the instructions to reset your password.
2. Open Snellius On Demand at [https://ondemand.snellius.surf.nl/](https://ondemand.snellius.surf.nl/).
3. Click on VSCode.
4. Set `gpu_h100` partition. Change time to 12:00:00. Set number of CPU cores to 16. Set memory to 32GB. Request 1 GPU. Click on "Launch".
5. Wait for the VSCode session to start. Once it is ready, click on "Connect to VSCode".
6. Click on the three horizontal lines in top left on the VSCode interface. Click on "Terminal" and then "New Terminal". This will open a terminal in the bottom of the VSCode interface.
7. Type `git clone https://github.com/jansam123/ML_project.git` to clone the repository. Then, type `cd ML_project` to navigate to the project directory.
8. You can either work from the home directory and see all the files, or it's nicer to reopen the VSCode interface in the `ML_project` directory. To do this, click on the three horizontal lines in top left on the VSCode interface. Click on "File" and then "Open Folder". Navigate to the `ML_project` directory and click "Open". This will open the project in the VSCode interface, and you can see all the files in the left sidebar. The path should like `/home/scur0034/ML_project`, where `scur0034` is your username.
9. You can then click on README.md to see these instructions

## Alternative: SSH access
1. With your login credentials, you can also access Snellius via SSH. Open a terminal on your local machine and type:
   ```bash
   ssh scur0034@snellius.surf.nl
   ```
   Change `scur0034` to your username. 
2. To enable SSH without entering your password every time, you can set up SSH keys. Follow the instructions [here](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660216/Connecting+to+the+system).
   


# Setting up conda environment

1. To properly use the correct and latest packages, we will install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and create a conda environment. 
2. Run this commands to install conda:
   ```bash
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p ~/conda
   echo "source  ~/conda/bin/activate" >> ~/.bashrc
   source ~/.bashrc
   rm Miniforge3-Linux-x86_64.sh
   ```
3. Create a conda environment with latest python:
    ```bash 
    conda create -n ml_project 
    ```
4. Activate the conda environment:
    ```bash
    conda activate ml_project
    ```
5. Install pytorch:
    ```bash
    conda install pytorch
    ```
6. Check GPU is available:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ``` 
5. Every time you start a new terminal, make sure to activate the conda environment with `conda activate ml_project` before running any code.


# Getting the dataset

The dataset is already downloaded on Snellius, and it's available at `/home/scur0034/ML_project/data`.
However, if you want to download the dataset yourself, you can use the `get_datasets.py` script available in the repository. Be aware this can take some time to download.
```bash
conda install requests tqdm
python get_datasets.py JetClass -d data -f
```
** The disk space on Snellius per user is limited to 200GB. You won't be able to download the entire dataset into your home directory.**


# Git Crash Course

> **Two ways to use git:** You can either use the terminal commands described below, or use the **built-in VSCode Git extension** (the branch icon in the left sidebar). Both do the same thing — the VSCode extension is just a graphical interface for the same git commands. See the [Using the VSCode Git Extension](#using-the-vscode-git-extension) section below if you prefer clicking over typing.

## What is Git?

Git is a version control system. Think of it like "Track Changes" in Google Docs, but for code. It keeps a full history of every change ever made, so you can always go back to a previous version. It also lets multiple people work on the same project without overwriting each other's work.

Key concepts:
- **Repository (repo):** The project folder, plus all the history of changes. You already have one — this `ML_project` folder is a git repo.
- **Commit:** A saved snapshot of your changes. Like a save point in a game — you can always return to it.
- **Branch:** A parallel version of the code. You create one, make changes there, and when you're happy with them, you merge it back into the main version.
- **Remote:** The version of the repo on GitHub (online). Your local copy and the remote can get out of sync — `push` sends your changes up, `pull` brings others' changes down.
- **Staging area:** A middle step before committing. You choose which changed files to include in the next commit (using `git add`). This lets you commit only the relevant files, not everything at once.

## Initial Setup

Before you can push to the repository, you need to tell git who you are. This name and email will appear next to your commits:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Since the repo uses HTTPS, GitHub will ask for authentication when you push. Your GitHub password won't work — you need a Personal Access Token (PAT):
1. Go to [GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens).
2. Click "Generate new token (classic)".
3. Give it a name, set an expiration, and check the `repo` scope.
4. Copy the token. When git asks for your password, paste this token instead of your GitHub password.

To avoid entering credentials every time you push or pull:
```bash
git config --global credential.helper store
```
After running this, the next time you enter your username and token, git will remember them.

## Essential Commands

### Check the current state
These commands don't change anything — they just show you information. Use them often.
```bash
git status          # shows which files are modified, staged, or untracked
git log --oneline   # shows the commit history (one line per commit)
git branch          # lists all local branches, stars the one you're on
```

### Pull latest changes from remote
This downloads any new commits others have pushed to GitHub and updates your local copy:
```bash
git pull
```
Always do this before you start working. If you and someone else both change the same file without pulling first, you'll get a merge conflict (annoying, but fixable).

### Create a new branch
A branch is an independent copy of the code where you can make changes without affecting `main`. Think of it as your own workspace:
```bash
git checkout -b my-feature-branch
```
This creates a new branch called `my-feature-branch` and switches you to it. Name it something descriptive, like `add-training-loop` or `fix-data-loading`.

### Stage and commit changes
Git doesn't automatically save your changes — you have to explicitly tell it what to save. This is a two-step process:

1. **Stage** — select which files to include in the commit:
```bash
git add file1.py file2.py   # stage specific files
git add .                    # stage ALL changed files (use with caution)
```

2. **Commit** — save those staged changes as a snapshot with a message:
```bash
git commit -m "Short description of what you changed"
```
The message should explain *what* you did and *why*, not just "changes" or "update".

### Push your branch to GitHub
Committing only saves changes locally (on Snellius). To upload them to GitHub so others can see:
```bash
git push -u origin my-feature-branch   # first time pushing this branch
git push                                # every time after that
```
The `-u origin my-feature-branch` part tells git "link this local branch to a remote branch of the same name". You only need it once per branch.

### Switch between branches
To jump between branches (e.g., to check something on `main` or go back to your branch):
```bash
git checkout main              # switch to main
git checkout my-feature-branch # switch back to your branch
```
Make sure you've committed or stashed your changes before switching, otherwise git might complain.

### Merge main into your branch (to stay up to date)
If others have pushed new code to `main` while you were working on your branch, you should pull those changes into your branch to stay up to date:
```bash
git checkout my-feature-branch
git pull origin main
```
If there are conflicts (you and someone else changed the same lines), git will tell you. See "Things to Avoid" below for how to handle them.

### Merge your branch into main (when your feature is done)
The preferred way is to open a Pull Request on GitHub — this lets others review your code before it goes into `main`. But if you want to do it locally:
```bash
git checkout main
git pull                       # get latest main first
git merge my-feature-branch    # bring your changes into main
git push                       # upload to GitHub
```

### Delete a branch after merging
Once your branch is merged, you don't need it anymore:
```bash
git branch -d my-feature-branch            # delete locally
git push origin --delete my-feature-branch  # delete on GitHub
```

## Typical Workflow (day-to-day)

Here's what a normal work session looks like:
```bash
git checkout main              # start from main
git pull                       # get latest changes
git checkout -b my-new-thing   # create your branch

# ... do your work, edit files ...

git status                     # see what you changed
git add file_you_changed.py    # stage the files
git commit -m "Add new thing"  # commit
git push -u origin my-new-thing  # push to GitHub
```

Then repeat the `add → commit → push` cycle as you keep working.

## Good Practices

1. **Commit often, commit small.** Each commit should represent one logical change. Don't bundle unrelated changes together — it makes it impossible to undo just one thing later.
2. **Write meaningful commit messages.** `"Fix learning rate bug in training loop"` is good. `"stuff"` or `"update"` is not. Future-you will thank present-you.
3. **Pull before you push.** This reduces merge conflicts.
4. **Use branches.** Keep `main` clean and working. Develop features on separate branches.
5. **Don't commit large files.** Git stores the full history of every file. A 2GB dataset committed once stays in the repo forever, even if you delete it later. Use `.gitignore` to exclude these.
6. **Review what you're committing.** Run `git status` and `git diff` before committing to make sure you're not accidentally including junk.

## `.gitignore`

The `.gitignore` file tells git which files to pretend don't exist. Any file matching a pattern in `.gitignore` won't show up in `git status` and won't be staged with `git add .`. This is essential for keeping large files, temporary files, and system junk out of the repo.

Create one in the project root if it doesn't exist:
```bash
touch .gitignore
```

Here's what you should put in it for this project:
```
# Data and model outputs (large files — never commit these)
data/
models/
checkpoints/
*.h5
*.hdf5
*.root
*.pt
*.onnx

# Python bytecode (auto-generated, not needed in repo)
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/

# Jupyter notebook checkpoints (auto-generated)
.ipynb_checkpoints/

# Environment and logs
.env
*.log

# OS junk (macOS and Windows create these automatically)
.DS_Store
Thumbs.db

# IDE settings (personal to each developer)
.vscode/
.idea/
```

After creating or editing `.gitignore`, commit it:
```bash
git add .gitignore
git commit -m "Add .gitignore"
git push
```

**Note:** `.gitignore` only prevents *untracked* files from being added. If a file is already tracked by git (was committed before), adding it to `.gitignore` won't remove it. To stop tracking a file that's already committed:
```bash
git rm --cached filename
git commit -m "Stop tracking filename"
```

## Things to Avoid

- **Don't push directly to `main`** without checking that your code works. Use a branch, test it, then merge.
- **Don't commit secrets** (passwords, API keys, tokens). If you accidentally do, changing the file isn't enough — the secret lives forever in git history. You'd need to rewrite history to truly remove it.
- **Don't commit data or large files.** Use `.gitignore`. If you accidentally committed a large file, ask for help removing it.
- **Don't use `git add .` blindly.** Always check `git status` first to see what's being staged. You might accidentally add data files, logs, or other garbage.
- **Don't force push (`git push -f`)** unless you really know what you're doing. It rewrites history on GitHub and can delete other people's work permanently.
- **Don't panic on merge conflicts.** They sound scary but they're just git saying "two people changed the same lines and I don't know which version to keep". Open the conflicting file, look for the `<<<<<<<`, `=======`, `>>>>>>>` markers, keep the code you want, delete the markers, then `git add` the file and `git commit`.

## Using the VSCode Git Extension

If you prefer a visual interface over typing commands, VSCode has a built-in Git extension that does everything described above — just with clicks instead of commands. You'll find it in the left sidebar (the icon that looks like a branching line). The initial setup (git config, PAT token) still needs to be done in the terminal, but after that you can do everything from the GUI.

### Viewing changes
Click the **Source Control** icon in the left sidebar (or press `Ctrl+Shift+G`). This shows you the same information as `git status` — which files were modified, added, or deleted. Click on any file to see a side-by-side diff (what changed).

### Staging files
In the Source Control panel, you'll see your changed files listed under "Changes". To stage a file (equivalent to `git add`):
- Hover over the file and click the **+** button to stage it.
- To stage all files, click the **+** next to "Changes".

Staged files move to the "Staged Changes" section. To unstage, click the **−** button.

### Committing
Once you've staged your files:
1. Type your commit message in the text box at the top of the Source Control panel.
2. Click the **checkmark** button (or press `Ctrl+Enter`).

This is the same as `git commit -m "your message"`.

### Pushing and pulling
After committing, click the **"..." menu** (three dots) at the top of the Source Control panel:
- **Push** — uploads your commits to GitHub.
- **Pull** — downloads new commits from GitHub.

Or use the sync button (circular arrows) in the bottom-left status bar — this does pull + push in one step.

### Branches
To create or switch branches, click the **branch name** in the bottom-left corner of VSCode (it says `main` or whatever branch you're on). A dropdown appears where you can:
- Select an existing branch to switch to it.
- Type a new name and select "Create new branch" to make one.

### Merge conflicts
When a merge conflict happens, VSCode highlights the conflicting sections in the file with colors and gives you buttons:
- **Accept Current Change** — keep your version.
- **Accept Incoming Change** — keep the other person's version.
- **Accept Both Changes** — keep both.

Pick the right one for each conflict, save the file, then stage and commit it.

### When to use the terminal anyway
The VSCode extension covers most daily tasks, but some things are easier or only possible in the terminal:
- `git log --oneline` — the terminal gives you a quick compact history.
- `git pull origin main` — pulling a specific branch into yours.
- `git push -u origin branch-name` — the first push of a new branch (VSCode may ask you to "publish" the branch, which does the same thing).
- Anything involving more advanced operations (rebase, cherry-pick, etc.)

You can mix and match — use the GUI for staging/committing/pushing and the terminal for everything else. They work on the same underlying git, so they don't conflict.
