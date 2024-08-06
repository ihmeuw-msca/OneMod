import functools
import subprocess

import tomllib

run = functools.partial(subprocess.run, shell=True)


def build_doc(version: str) -> None:
    print(f"Build _{version}_")
    run(f"git checkout v{version}")
    run("git checkout publish-docs -- conf.py")
    run("git checkout publish-docs -- versions.toml")

    #run("make html")
    run("sphinx-build -M html . ../_build")
    run("ls _build")
    run(f"mv _build/html pages/{version}")
    run("rm -rf _build")
    run("git checkout publish-docs")


def build_init_page(version: str) -> None:
    with open("pages/index.html", "w") as f:
        f.write(f"""<!doctype html>
<meta http-equiv="refresh" content="0; url=./{version}/">""")


if __name__ == "__main__":
    # create pages folder
    print("Python main")
    run("rm -rf pages")
    run("mkdir pages")

    # get versions
    with open("meta.toml", "rb") as f:
        versions = tomllib.load(f)["versions"]
    print(f"versions A _{versions}")
    versions.sort(reverse=True, key=lambda v: tuple(map(int, v.split("."))))
    print(f"versions B _{versions}_")

    # build documentations for different versions
    for version in versions:
        build_doc(version)

    # build initial page that redirect to the latest version
    build_init_page(versions[0])
