import tomli


def load_config(path: str):
    return tomli.load(open(path, "rb"))
