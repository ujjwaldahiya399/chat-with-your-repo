# another new line
# a new comment line
import functools


CONSTANT = 42


@functools.lru_cache
def cached_func():
    return 1


@app.route("/login")
@require_auth
def login():
    return "ok"


class Handler:
    pass
