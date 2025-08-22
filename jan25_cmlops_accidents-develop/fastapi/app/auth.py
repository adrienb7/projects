from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from config import settings

# Template
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

# Check API_KEY
async def get_api_key(api_key_header: str = Security(api_key_header)):
    #
    if api_key_header in settings.API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Accés non autorisé"
    )
