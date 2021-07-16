#!/bin/sh

ProgName=$(basename $0)

execute() {
    Action=$1
    Target=$2
    shift 2
    if [[ -z $Action ]]; then
        echo "Action is missing: check 'help' for more info"
        return 1
    fi
    if [[ -z $Target ]]; then
        echo "Target is missing: please specify one of [dev|prod]"
        return 1
    fi
    echo "Setting $Target environment..."
    # save old env path and link new one
    OldEnv=$(readlink .env)
    ln -sf env/$Target.env .env
    # set project name variable
    ProjectName="importer-$Target"
    ProjectName=${ProjectName%"-prod"}
    # execute docker-compose commands
    echo "Executing action '$Action' for target: $BUILD_TARGET"
    docker-compose -p $ProjectName \
        -f docker-compose.yml \
        -f docker-compose.$Target.yml $Action $@
    # reset old env once done
    echo "Restoring environment $OldEnv..."
    ln -sf $OldEnv .env
}

cmd_help() {
    echo "Usage: $ProgName (help|lint|test|build|run|stop|down|db)"
    echo "lint              executes a global linting of the python code using flake8 (exec from environment)"
    echo "test              launches tests using docker compose"
    echo "test-local        launches tests locally, using docker compose containers as support"
    echo "build  [dev|prod] [docker params]  builds development or production containers"
    echo "run    [dev|prod] [docker params]  runs development or production containers"
    echo "stop   [dev|prod] [docker params]  stops development or production containers"
    echo "down   [dev|prod] [docker params]  kills development or production containers"
    echo "config [dev|prod] [docker params]  prints the final configuratoin, for development, production, or test"
    echo "help              prints the current message"
    echo ""
}

cmd_lint() {
    echo "Checking for errors or warnings..."
    flake8 --max-line-length=120 --verbose serve/
}

cmd_build() {
    echo "Building images..."
    Target=$1
    shift
    execute build $Target $@
}

cmd_run() {
    echo "Running project..."
    Target=$1
    shift
    execute up $Target $@
}

cmd_stop() {
    echo "Stopping containers..."
    Target=$1
    shift
    execute stop $Target $@
}

cmd_down() {
    echo "Killing containers..."
    Target=$1
    shift
    execute down $Target $@
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
