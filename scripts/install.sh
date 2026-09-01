#!/bin/sh

set -eu

REPOSITORY_RELEASES_URL=${_BETTERBORG_RELEASES_URL:-https://github.com/betterborg/betterborg-cli/releases}
SELECTED_VERSION=${BETTERBORG_VERSION:-}
INSTALL_HOME=${HOME:-}
TEMPORARY_DIRECTORY=
STAGED_EXECUTABLE=

cleanup() {
    if [ -n "$STAGED_EXECUTABLE" ]; then
        rm -f "$STAGED_EXECUTABLE"
    fi
    if [ -n "$TEMPORARY_DIRECTORY" ]; then
        rm -rf "$TEMPORARY_DIRECTORY"
    fi
}

trap cleanup EXIT
trap 'exit 1' HUP INT TERM

fail() {
    printf 'betterborg installer: %s\n' "$1" >&2
    exit 1
}

unsupported() {
    printf 'betterborg installer: %s\n' "$1" >&2
    printf '%s\n' \
        'Fallback: install uv, then run `uvx --from betterborg betterborg version`.' >&2
    exit 1
}

download() {
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --fail --location --silent --show-error --output "$2" "$1"
}

manifest_version() {
    awk '
        $0 == "  \"schema_version\": 1," { schemas += 1 }
        index($0, "  \"version\": \"") == 1 {
            value = $0
            sub(/^  "version": "/, "", value)
            sub(/",$/, "", value)
            versions += 1
        }
        END {
            if (schemas == 1 && versions == 1) {
                print value
            } else {
                exit 1
            }
        }
    ' "$1"
}

manifest_checksum() {
    awk -v filename="$2" -v operating_system="$3" -v architecture="$4" '
        $0 == "      \"filename\": \"" filename "\"," {
            active = 1
            correct_os = 0
            correct_arch = 0
            digest = ""
            next
        }
        active && $0 == "      \"os\": \"" operating_system "\"," {
            correct_os = 1
            next
        }
        active && $0 == "      \"arch\": \"" architecture "\"," {
            correct_arch = 1
            next
        }
        active && index($0, "      \"sha256\": \"") == 1 {
            digest = $0
            sub(/^      "sha256": "/, "", digest)
            sub(/",$/, "", digest)
            next
        }
        active && ($0 == "    }," || $0 == "    }") {
            if (correct_os && correct_arch && digest != "") {
                selected = digest
                matches += 1
            }
            active = 0
        }
        END {
            if (matches == 1) {
                print selected
            } else {
                exit 1
            }
        }
    ' "$1"
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{ print $1 }'
    else
        fail 'SHA-256 verification requires sha256sum or shasum.'
    fi
}

detect_target() {
    detected_system=$(uname -s 2>/dev/null || printf unknown)
    detected_architecture=$(uname -m 2>/dev/null || printf unknown)

    case "$detected_system" in
        Linux)
            kernel_release=$(uname -r 2>/dev/null || printf unknown)
            case "$kernel_release" in
                *[Mm]icrosoft*)
                    case "$kernel_release" in
                        *WSL2*|*wsl2*) ;;
                        *)
                            unsupported \
                                'Windows installation is supported only inside WSL2; upgrade this WSL environment to WSL2 and retry.'
                            ;;
                    esac
                    ;;
            esac
            TARGET_OS=linux
            ;;
        Darwin)
            TARGET_OS=darwin
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows*)
            unsupported \
                'Native Windows is unsupported; open a WSL2 shell and run the installer there.'
            ;;
        *)
            unsupported \
                "No standalone Betterborg release supports $detected_system/$detected_architecture."
            ;;
    esac

    case "$detected_architecture" in
        x86_64|amd64) TARGET_ARCH=x86_64 ;;
        arm64|aarch64) TARGET_ARCH=arm64 ;;
        *)
            unsupported \
                "No standalone Betterborg release supports $detected_system/$detected_architecture."
            ;;
    esac
    TARGET="betterborg-$TARGET_OS-$TARGET_ARCH"
}

detect_target

command -v curl >/dev/null 2>&1 || fail 'curl is required to download Betterborg.'
[ -n "$INSTALL_HOME" ] || fail 'HOME must be set for a persistent user installation.'

TEMPORARY_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/betterborg-install.XXXXXXXX") || \
    fail 'could not create a temporary download directory.'
MANIFEST="$TEMPORARY_DIRECTORY/release-manifest.json"
ARTIFACT="$TEMPORARY_DIRECTORY/$TARGET"

if [ -n "$SELECTED_VERSION" ]; then
    case "$SELECTED_VERSION" in
        [0-9]*) ;;
        *) fail 'BETTERBORG_VERSION is invalid.' ;;
    esac
    case "$SELECTED_VERSION" in
        *[!0-9A-Za-z._+-]*) fail 'BETTERBORG_VERSION is invalid.' ;;
    esac
    MANIFEST_URL="$REPOSITORY_RELEASES_URL/download/v$SELECTED_VERSION/release-manifest.json"
else
    MANIFEST_URL="$REPOSITORY_RELEASES_URL/latest/download/release-manifest.json"
fi

download "$MANIFEST_URL" "$MANIFEST" || fail 'could not download the release manifest.'

VERSION=$(manifest_version "$MANIFEST") || fail 'release manifest metadata is invalid.'
case "$VERSION" in
    [0-9]*) ;;
    *) fail 'release manifest version is invalid.' ;;
esac
case "$VERSION" in
    *[!0-9A-Za-z._+-]*) fail 'release manifest version is invalid.' ;;
esac
[ -z "$SELECTED_VERSION" ] || [ "$VERSION" = "$SELECTED_VERSION" ] || \
    fail "release manifest reported $VERSION, expected $SELECTED_VERSION."
EXPECTED_SHA256=$(manifest_checksum "$MANIFEST" "$TARGET" "$TARGET_OS" "$TARGET_ARCH") || \
    fail "release manifest does not contain the supported target $TARGET."
case "$EXPECTED_SHA256" in
    *[!0-9a-f]*|'') fail "release manifest checksum for $TARGET is invalid." ;;
esac
[ "${#EXPECTED_SHA256}" -eq 64 ] || \
    fail "release manifest checksum for $TARGET is invalid."

VERSIONED_RELEASE_URL="$REPOSITORY_RELEASES_URL/download/v$VERSION"
download "$VERSIONED_RELEASE_URL/$TARGET" "$ARTIFACT" || \
    fail "could not download $TARGET for Betterborg $VERSION."
ACTUAL_SHA256=$(sha256 "$ARTIFACT")
[ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ] || \
    fail "downloaded $TARGET failed SHA-256 verification; nothing was installed."

INSTALL_DIRECTORY="$INSTALL_HOME/.local/bin"
INSTALL_PATH="$INSTALL_DIRECTORY/betterborg"
mkdir -p "$INSTALL_DIRECTORY" || fail "could not create $INSTALL_DIRECTORY."
if [ -e "$INSTALL_PATH" ] && [ ! -f "$INSTALL_PATH" ]; then
    fail "$INSTALL_PATH exists and is not a regular file."
fi

STAGED_EXECUTABLE=$(mktemp "$INSTALL_DIRECTORY/.betterborg.install.XXXXXXXX") || \
    fail "could not stage the executable in $INSTALL_DIRECTORY."
cp "$ARTIFACT" "$STAGED_EXECUTABLE" || fail 'could not stage the verified executable.'
chmod 755 "$STAGED_EXECUTABLE" || fail 'could not make the staged executable runnable.'

if ! STAGED_VERSION=$("$STAGED_EXECUTABLE" version 2>/dev/null); then
    fail 'the verified download could not report its version; nothing was installed.'
fi
[ "$STAGED_VERSION" = "betterborg $VERSION" ] || \
    fail "the verified download reported '$STAGED_VERSION', expected 'betterborg $VERSION'; nothing was installed."

mv -f "$STAGED_EXECUTABLE" "$INSTALL_PATH" || \
    fail "could not atomically install $INSTALL_PATH."
STAGED_EXECUTABLE=

if ! INSTALLED_VERSION=$("$INSTALL_PATH" version 2>/dev/null); then
    fail "the persistent executable at $INSTALL_PATH failed verification."
fi
[ "$INSTALLED_VERSION" = "betterborg $VERSION" ] || \
    fail "the persistent executable at $INSTALL_PATH failed version verification."

printf 'Installed Betterborg %s at %s.\n' "$VERSION" "$INSTALL_PATH"
case ":${PATH:-}:" in
    *":$INSTALL_DIRECTORY:"*) ;;
    *)
        printf '%s\n' \
            "Add $INSTALL_DIRECTORY to PATH, then restart open terminals and plugin hosts."
        printf '%s\n' \
            "For POSIX shells: export PATH=\"$INSTALL_DIRECTORY:\$PATH\""
        ;;
esac

if ! PATH="$INSTALL_DIRECTORY${PATH:+:$PATH}" \
    "$INSTALL_PATH" plugins install --all; then
    fail 'the CLI was installed, but host plugin activation did not complete.'
fi

printf '%s\n' 'Betterborg CLI verification and plugin activation completed.'
