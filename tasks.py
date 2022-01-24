"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Author: Shyue Ping Ong
"""

from invoke import task
import os
import json
import requests
import re
import subprocess

from camd import __version__ as CURRENT_VER
from camd.utils import get_new_version

NEW_VER = get_new_version(CURRENT_VER)
EDITOR = os.environ.get("EDITOR", "vim")


@task
def publish(ctx):
    """
    Upload release to Pypi using twine.

    :param ctx:
    """
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def set_ver(ctx):
    lines = []
    with open("camd/__init__.py", "rt") as f:
        for l in f:
            if l.startswith("__version__"):
                lines.append('__version__ = "%s"' % NEW_VER)
            else:
                lines.append(l.rstrip())
    with open("camd/__init__.py", "wt") as f:
        f.write("\n".join(lines))
        f.write("\n")

    lines = []
    with open("setup.py", "rt") as f:
        for l in f:
            lines.append(re.sub(r'version=([^,]+),', 'version="%s",' % NEW_VER,
                                l.rstrip()))
    with open("setup.py", "wt") as f:
        f.write("\n".join(lines))
        f.write("\n")


@task
def tag_release(ctx):
    """
    Tag and merge into stable branch.

    :param ctx:
    """
    ctx.run("git commit -a -m \"v%s release\"" % (NEW_VER, ), warn=True)
    ctx.run("git tag -a v%s -m \"v%s release\"" % (NEW_VER, NEW_VER))
    ctx.run("git push --tags")


@task
def release_github(ctx):
    """
    Release to Github using Github API.

    :param ctx:
    """
    with open("CHANGES.md") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "main",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/TRI-AMDD/camd/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def update_changelog(ctx):
    """
    Create a preliminary change log using the git logs.

    :param ctx:
    """
    output = subprocess.check_output(["git", "log", "--pretty=format:%s",
                                      "v%s..HEAD" % CURRENT_VER])
    lines = ["* " + l for l in output.decode("utf-8").strip().split("\n")]
    with open("CHANGES.md") as f:
        contents = f.read()
    l = "=========="
    toks = contents.split(l)
    head = "\n\nv%s\n" % NEW_VER + "-" * (len(NEW_VER) + 1) + "\n"
    toks.insert(-1, head + "\n".join(lines))
    with open("CHANGES.md", "w") as f:
        f.write(toks[0] + l + "".join(toks[1:]))
    ctx.run("open CHANGES.md")

@task
def check_creds(ctx):
    req_env_vars = ["TWINE_USERNAME", "TWINE_PASSWORD", "GITHUB_RELEASES_TOKEN"]
    for env_var in req_env_vars:
        assert os.environ.get(env_var), "{} is not set".format(env_var)


@task
def release(ctx, notest=False, nover=False):
    """
    Run full sequence for releasing camd.

    :param ctx:
    :param notest: Whether to skip tests.
    :param notest: Whether to skip autoversion (e. g. if tagging version).
    """
    check_creds(ctx)
    ctx.run("rm -r dist build camd.egg-info", warn=True)
    if not nover:
        set_ver(ctx)
    if not notest:
        ctx.run("pytest camd")
    publish(ctx)
    tag_release(ctx)
    release_github(ctx)
