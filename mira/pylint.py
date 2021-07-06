import pylint.lint.pylinter


def patch(paths):
    """Patch pylint so that modules with paths starting with
    any of a list of paths are ignored."""
    original_expand_modules = pylint.lint.pylinter.expand_modules

    def expand_modules(*args, **kwargs):
        """Patched expand_module that excludes undesired paths."""
        result, errors = original_expand_modules(*args, **kwargs)
        result = list(
            filter(
                lambda item: not any(item["path"].startswith(path) for path in paths),
                result,
            )
        )

        return result, errors

    pylint.lint.pylinter.expand_modules = expand_modules
