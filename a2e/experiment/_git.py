import git


def git_hash():
    """Returns the current hash of this git repository"""
    repo = git.Repo(search_parent_directories=True)
    current_hash = repo.head.object.hexsha

    return current_hash


def git_diff():
    """Returns the current diff of this git repository compared to HEAD"""
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff('HEAD')

    return diff
