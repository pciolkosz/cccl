#!/bin/bash

# This script parses the CI matrix.yaml file and generates a devcontainer.json file for each unique combination of
# CUDA version, compiler name/version, and Ubuntu version. The devcontainer.json files are written to the
# .devcontainer directory to a subdirectory named after the CUDA version and compiler name/version.
# GitHub docs on using multiple devcontainer.json files:
# https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers#devcontainerjson

set -euo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

function usage {
    echo "Usage: $0 [--clean] [-h/--help] [-v/--verbose]"
    echo "  --clean   Remove stale devcontainer subdirectories"
    echo "  -h, --help   Display this help message"
    echo "  -v, --verbose  Enable verbose mode (set -x)"
    exit 1
}

# Function to update the devcontainer.json file with the provided parameters
update_devcontainer() {
    local input_file="$1"
    local output_file="$2"
    local name="$3"
    local cuda_version="$4"
    local cuda_ext="$5"
    local compiler_name="$6"
    local compiler_exe="$7"
    local compiler_version="$8"
    local devcontainer_version="$9"
    local internal="${10}"

    local cuda_suffix=""
    if $cuda_ext; then
        local cuda_suffix="ext"
    fi

    # NVHPC SDK comes with its own bundled toolkit
    local toolkit_name="-cuda${cuda_version}${cuda_suffix}"
    if [ $compiler_name == "nvhpc" ]; then
        toolkit_name=""
    fi

    local IMAGE_ROOT="rapidsai/devcontainers:${devcontainer_version}-cpp-"
    local INTERNAL_ROOT="gitlab-master.nvidia.com:5005/cccl/cccl-devcontainers:cpp-"

    img=$IMAGE_ROOT
    if [ "$internal" == "true" ]; then
        img=$INTERNAL_ROOT
    fi;

    local image="${img}${compiler_name}${compiler_version}${toolkit_name}"

    jq --arg image "$image" \
       --arg name "$name" \
       --arg cuda_version "$cuda_version" \
       --arg cuda_ext "$cuda_ext" \
       --arg compiler_name "$compiler_name" \
       --arg compiler_exe "$compiler_exe" \
       --arg compiler_version "$compiler_version" \
       '.image = $image |
        .name = $name |
        .containerEnv.DEVCONTAINER_NAME = $name |
        .containerEnv.CCCL_BUILD_INFIX = $name |
        .containerEnv.CCCL_CUDA_VERSION = $cuda_version |
        .containerEnv.CCCL_CUDA_EXTENDED = $cuda_ext |
        .containerEnv.CCCL_HOST_COMPILER = $compiler_name |
        .containerEnv.CCCL_HOST_COMPILER_VERSION = $compiler_version '\
       "$input_file" > "$output_file"
}

make_name() {
    local cuda_version="$1"
    local cuda_ext="$2"
    local compiler_name="$3"
    local compiler_version="$4"

    local cuda_suffix=""
    if $cuda_ext; then
        local cuda_suffix="ext"
    fi

    echo "cuda${cuda_version}${cuda_suffix}-${compiler_name}${compiler_version}"
}

CLEAN=false
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            ;;
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        *)
            usage
            ;;
    esac
    shift
done

MATRIX_FILE="../ci/matrix.yaml"
COMPUTE_MATRIX="../.github/actions/workflow-build/build-workflow.py"

# Enable verbose mode if requested
if [ "$VERBOSE" = true ]; then
    set -x
    cat ${MATRIX_FILE}
fi

# Read matrix.yaml and convert it to json
matrix_json=$(python3 ${COMPUTE_MATRIX} ${MATRIX_FILE} --devcontainer-info)

if [ "$VERBOSE" = true ]; then
    echo "$matrix_json"
fi

# Get the devcontainer image version and define image tag root
readonly DEVCONTAINER_VERSION=$(echo "$matrix_json" | jq -r '.devcontainer_version')

# Get unique combinations of cuda version, compiler name/version, and Ubuntu version
readonly combinations=$(echo "$matrix_json" | jq -c '.combinations[]')

# Update the base devcontainer with the default values
# The root devcontainer.json file is used as the default container as well as a template for all
# other devcontainer.json files by replacing the `image:` field with the appropriate image name
readonly base_devcontainer_file="./devcontainer.json"
readonly NEWEST_GCC_CUDA_ENTRY=$(echo "$combinations" | jq -rs '[.[] | select(.compiler_name == "gcc")] | sort_by((.cuda | tonumber), (.compiler_version | tonumber)) | .[-1]')
readonly NEWEST_LLVM_CUDA_ENTRY=$(echo "$combinations" | jq -rs '[.[] | select(.compiler_name == "llvm")] | sort_by((.cuda | tonumber), (.compiler_version | tonumber)) | .[-1]')
readonly DEFAULT_CUDA=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.cuda')
readonly DEFAULT_CUDA_EXT=false
readonly DEFAULT_COMPILER_NAME=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_name')
readonly DEFAULT_COMPILER_EXE=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_exe')
readonly DEFAULT_COMPILER_VERSION=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_version')
readonly DEFAULT_NAME=$(make_name "$DEFAULT_CUDA" "$DEFAULT_CUDA_EXT" "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_VERSION")

update_devcontainer ${base_devcontainer_file} "./temp_devcontainer.json" "$DEFAULT_NAME" "$DEFAULT_CUDA" "$DEFAULT_CUDA_EXT" "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_EXE" "$DEFAULT_COMPILER_VERSION" "$DEVCONTAINER_VERSION" "false"
mv "./temp_devcontainer.json" ${base_devcontainer_file}

# Always create an extended version of the default devcontainer:
readonly EXT_NAME=$(make_name "$DEFAULT_CUDA" true "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_VERSION")
update_devcontainer ${base_devcontainer_file} "./temp_devcontainer.json" "$EXT_NAME" "$DEFAULT_CUDA" true "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_EXE" "$DEFAULT_COMPILER_VERSION" "$DEVCONTAINER_VERSION" "false"
mkdir -p "$EXT_NAME"
mv "./temp_devcontainer.json" "$EXT_NAME/devcontainer.json"


# Create an array to keep track of valid subdirectory names
valid_subdirs=("$EXT_NAME")

# The img folder should not be removed:
valid_subdirs+=("img")

# Don't remove RAPIDS containers:
for rapids_container in *rapids*; do
    valid_subdirs+=("${rapids_container}")
done

# Inject ctk version 99.9
readonly cuda99_9_gcc=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -rsc '.[].cuda |= "99.9" | .[].internal |= true | .[-1]')
readonly cuda99_8_gcc=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -rsc '.[].cuda |= "99.8" | .[].internal |= true | .[-1]')
readonly cuda99_9_llvm=$(echo "$NEWEST_LLVM_CUDA_ENTRY" | jq -rsc '.[].cuda |= "99.9" | .[].internal |= true | .[-1]')
readonly cuda99_8_llvm=$(echo "$NEWEST_LLVM_CUDA_ENTRY" | jq -rsc '.[].cuda |= "99.8" | .[].internal |= true | .[-1]')

readonly all_comb="$combinations $cuda99_9_gcc $cuda99_8_gcc $cuda99_9_llvm $cuda99_8_llvm"
# For each unique combination
for combination in $all_comb; do
    cuda_version=$(echo "$combination" | jq -r '.cuda')
    cuda_ext=$(echo "$combination" | jq -r '.cuda_ext')
    compiler_name=$(echo "$combination" | jq -r '.compiler_name')
    compiler_exe=$(echo "$combination" | jq -r '.compiler_exe')
    compiler_version=$(echo "$combination" | jq -r '.compiler_version')
    internal=$(echo "$combination" | jq -r '.internal')

    name=$(make_name "$cuda_version" "$cuda_ext" "$compiler_name" "$compiler_version")
    mkdir -p "$name"
    new_devcontainer_file="$name/devcontainer.json"

    update_devcontainer "$base_devcontainer_file" "$new_devcontainer_file" "$name" "$cuda_version" "$cuda_ext" "$compiler_name" "$compiler_exe" "$compiler_version" "$DEVCONTAINER_VERSION" "$internal"
    echo "Created $new_devcontainer_file"

    # Add the subdirectory name to the valid_subdirs array
    valid_subdirs+=("$name")
done

# Clean up stale subdirectories and devcontainer.json files
if [ "$CLEAN" = true ]; then
    for subdir in ./*; do
        if [ -d "$subdir" ] && [[ ! " ${valid_subdirs[@]} " =~ " ${subdir#./} " ]]; then
            echo "Removing stale subdirectory: $subdir"
            rm -r "$subdir"
        fi
    done
fi
