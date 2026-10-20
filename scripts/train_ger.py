import argparse
from functools import partial
import inspect
import json
import fire
from vanilla_ger.ger import train
from vanilla_ger.common import resolve_path


def main():
    """Read --config, then preserve Fire's typed training flags and help."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    args, remaining = parser.parse_known_args()
    target = train
    if args.config:
        parameters = json.loads(resolve_path(args.config).read_text())
        if not isinstance(parameters, dict):
            raise ValueError("Training config must contain a JSON object")
        inspect.signature(train).bind_partial(**parameters)
        target = partial(train, **parameters)
    fire.Fire(target, command=remaining)

if __name__ == "__main__":
    main()
