#!/bin/sh

ProgName=$(basename $0)

cmd_help() {
    echo "Usage: $ProgName (help|importer)"
}

cmd_webserver() {
    uvicorn phenoai.asgi:app --host 0.0.0.0 --port 5000
    exit $?
}

subcommand=$1
case ${subcommand} in
"" | "-h" | "--help")
    cmd_help
    ;;
*)
    shift
    cmd_${subcommand} $@
    echo "Done!"
    if [[ $? == 127 ]]; then
        echo "Error: '$subcommand' is not a known subcommand." >&2
        echo "       Run '$(basename $0) --help' for a list of known subcommands." >&2
        exit 1
    fi
    ;;
esac
