import git


def git_hash():
    repo = git.Repo(search_parent_directories=True)
    current_hash = repo.head.object.hexsha

    return current_hash


def git_diff():
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff('HEAD')

    return diff
