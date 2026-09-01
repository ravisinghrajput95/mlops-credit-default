"""Authentication for the inference API.

A credit decision is a regulated act. Who asked for it is part of the record,
which is why this module does two jobs rather than one: it decides whether a
request is allowed, and it says *which caller* made it, so the caller lands in
the prediction log next to the score and the reasons. An unattributable decision
is a gap in the same audit trail the adverse-action reasons exist to fill.

Keys are presented as ``Authorization: Bearer <key>`` and configured as
``name:secret`` pairs, so revoking one caller does not disturb the others and the
log says "batch-scoring" rather than "some valid key".

Four decisions worth stating:

**Misconfiguration fails closed.** If authentication is demanded and no keys are
configured, the process refuses to start. This is the opposite of how the model
loader behaves -- a missing model degrades ``/health`` and keeps serving, because
an API reporting its own illness is more useful than a crash loop. A missing key
list cannot degrade the same way: "no keys configured" would otherwise mean
"nobody can be rejected", and an endpoint that quietly stops checking is worse
than one that is plainly down.

**Invalid keys get 401, not 403.** 403 means "we know who you are and you may
not"; there is no authorisation layer here, so every rejection is a failure to
authenticate. Returning 403 for a bad key is a common bug that misreports the
failure to the caller and to whoever is reading the logs.

**Comparison is constant-time.** ``secrets.compare_digest`` rather than ``==``,
so response timing does not leak how much of a key was correct. Cheap, and the
kind of thing that is embarrassing to be missing rather than impressive to have.

**Keys never reach a log.** They are held in a ``SecretStr`` so an accidental
``repr(settings)`` -- in a traceback, a debug print, a Prefect run log -- prints
``**********`` instead of the credential.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Short keys are guessable and there is no reason to permit one. 24 characters of
# base64 is ~144 bits; this floor only exists to stop "test" reaching production.
MIN_KEY_LENGTH = 24

UNAUTHENTICATED_WARNING = (
    "AUTHENTICATION IS OFF. /predict and /model-info are open to anyone who can "
    "reach this port. Set REQUIRE_AUTH=true and API_KEYS before exposing it."
)

# Bearer rather than a bespoke header: it is the standard, and FastAPI renders an
# Authorize button in the docs for it, so the demo is usable without curl.
# auto_error=False so this module owns the 401 and its WWW-Authenticate header
# rather than inheriting FastAPI's, which omits the challenge.
bearer_scheme = HTTPBearer(auto_error=False, description="API key issued to your service.")


@dataclass(frozen=True)
class ApiKey:
    """One caller's credential. ``name`` is an identity, not a secret."""

    name: str
    secret: str


def parse_api_keys(raw: str) -> list[ApiKey]:
    """Parse ``name:secret,name2:secret2`` into keys.

    A bare secret with no name is accepted and labelled ``unnamed``, because
    refusing to start over a missing label would be a poor trade -- but the log
    line then identifies the caller no better than "someone with a valid key",
    which is the reason to bother naming them.
    """
    keys: list[ApiKey] = []
    seen: set[str] = set()

    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, secret = entry.partition(":")
        if not separator:
            name, secret = "unnamed", entry
        name, secret = name.strip(), secret.strip()

        if len(secret) < MIN_KEY_LENGTH:
            raise ValueError(
                f"API key '{name}' is {len(secret)} characters; the minimum is "
                f"{MIN_KEY_LENGTH}. Generate one with: python -c "
                "'import secrets; print(secrets.token_urlsafe(32))'"
            )
        if name in seen:
            raise ValueError(f"Duplicate API key name '{name}'; names identify callers in the log.")
        seen.add(name)
        keys.append(ApiKey(name, secret))

    return keys


class Authenticator:
    """Resolves a presented credential to a caller name."""

    def __init__(self, keys: list[ApiKey], required: bool) -> None:
        if required and not keys:
            raise ValueError(
                "REQUIRE_AUTH is set but no API_KEYS are configured. Refusing to start: "
                "with no keys every request would be rejected, and a deployment that "
                "cannot authenticate anyone must fail loudly rather than look healthy."
            )
        self._keys = keys
        self.required = required

    @property
    def configured_callers(self) -> list[str]:
        """Names only. Used for start-up logging, and safe to print."""
        return [key.name for key in self._keys]

    def identify(self, presented: str | None) -> str:
        """Return the caller's name, or raise 401.

        ``anonymous`` when authentication is switched off, so the prediction log
        has a value in the caller column either way and downstream code never has
        to special-case a null.
        """
        if not self.required:
            return "anonymous"

        if not presented:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="An API key is required. Send it as: Authorization: Bearer <key>.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Every key is compared, and always all of them: returning early on the
        # first match would make response time depend on a key's position in the
        # list, which is the same leak compare_digest exists to close.
        matched: str | None = None
        for key in self._keys:
            if secrets.compare_digest(presented, key.secret):
                matched = key.name

        if matched is None:
            # Deliberately no echo of the presented value: a rejected credential
            # is still a credential, and log aggregators are not secret stores.
            logger.warning("Rejected a request presenting an unrecognised API key.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The API key presented is not recognised.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return matched


def build_authenticator(raw_keys: str, required: bool) -> Authenticator:
    """Construct from configuration. Raises rather than degrading -- see the module docstring."""
    authenticator = Authenticator(parse_api_keys(raw_keys), required)
    if required:
        logger.info(
            "Authentication is on; %d caller(s) configured: %s",
            len(authenticator.configured_callers),
            ", ".join(authenticator.configured_callers),
        )
    else:
        logger.warning(UNAUTHENTICATED_WARNING)
    return authenticator


def credentials_secret(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    """Pull the raw token out of the parsed Authorization header."""
    return credentials.credentials if credentials else None
