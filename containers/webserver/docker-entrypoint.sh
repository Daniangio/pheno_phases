#!/bin/sh

ProgName=$(basename $0)

cmd_help() {
    echo "Usage: $ProgName (help|importer)"
}

cmd_webserver() {
    uvicorn phenoai.asgi:app --host 0.0.0.0 --port 5000
    exit $?
}

cmd_test() {
    # Launches a PyTest session inside the container.
    # APP_WORKER:   always false, since we want to test through FastAPI
    # APP_TESTING:  boolean for customization in case of tests
    echo "Running pytest..."
    export APP_TESTING=true
    exec pytest --cov-report=term --cov=phenoai tests/
    return $?
}

subcommand=$1
case ${subcommand} in
"" | "-h" | "--help")
    cmd_help
    ;;
*)
    shift
    cmd_${subcommand} $@
    Return=$?
    if [ Return = 127 ]; then
        echo "Error: '$subcommand' is not a known subcommand." >&2
        echo "       Run '$(basename $0) --help' for a list of known subcommands." >&2
        exit 1
    fi
    if [ Return = 0 ]; then
        echo "Done!"
    fi
    ;;
esac