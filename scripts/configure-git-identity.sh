#!/bin/sh
# Run once in each clone. No global Git or gh account configuration is changed.
set -eu
root=$(git rev-parse --show-toplevel)
git config --local user.name mendrik-private
git config --local user.email 262438142+mendrik-private@users.noreply.github.com
git config --local credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper "!\"$root/scripts/git-credential-mendrik-private\""
git config --local credential.https://github.com.username mendrik-private
