"""Enter an owned process group before replacing this shim with the command."""
from __future__ import annotations

import os
import sys


if __name__ == '__main__':
    os.environ['CODELAW_PROCESS_GROUP'] = str(os.getpgrp())
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
