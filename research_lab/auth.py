import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().backend_service_token.get_secret_value()
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid service credentials")
