"""
Temporary debug middleware for diagnosing production auth issues.
Remove this file once auth is working correctly.
"""
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import jwt

logger = logging.getLogger(__name__)


class AuthDebugMiddleware(BaseHTTPMiddleware):
    """
    Debug middleware to log all auth-related information.
    This should be added BEFORE JWTAuthMiddleware in the middleware stack.
    Gets JWT_SECRET directly from environment to avoid startup crashes.
    """
    
    def __init__(self, app, jwt_secret: str = None):
        super().__init__(app)
        # Try to get secret from parameter, then fallback to environment
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET") or ""
        if not self.jwt_secret:
            logger.warning("⚠️ AuthDebugMiddleware: No JWT_SECRET available - token verification will fail")
    
    async def dispatch(self, request: Request, call_next):
        # Only log for protected endpoints
        if request.url.path.startswith("/api/v1/"):
            logger.info("=" * 80)
            logger.info(f"🔍 AUTH DEBUG for {request.method} {request.url.path}")
            logger.info("=" * 80)
            
            # 1. Check all possible auth headers (case-insensitive)
            logger.info("📋 All headers:")
            for header_name, header_value in request.headers.items():
                if 'auth' in header_name.lower() or 'jwt' in header_name.lower():
                    # Mask the token value for security
                    masked_value = header_value[:20] + "..." if len(header_value) > 20 else header_value
                    logger.info(f"   {header_name}: {masked_value}")
            
            # 2. Specific JWT headers
            access_jwt = request.headers.get("X-ACCESS-JWT") or request.headers.get("x-access-jwt")
            refresh_jwt = request.headers.get("X-REFRESH-JWT") or request.headers.get("x-refresh-jwt")
            authorization = request.headers.get("Authorization") or request.headers.get("authorization")
            
            logger.info(f"🔑 X-ACCESS-JWT present: {bool(access_jwt)}")
            logger.info(f"🔑 X-REFRESH-JWT present: {bool(refresh_jwt)}")
            logger.info(f"🔑 Authorization header present: {bool(authorization)}")
            
            # 3. JWT Secret info (length only, never log actual secret!)
            logger.info(f"🔐 JWT_SECRET configured: {bool(self.jwt_secret)}")
            logger.info(f"🔐 JWT_SECRET length: {len(self.jwt_secret) if self.jwt_secret else 0}")
            
            # 4. Try to decode the token without verification to see payload
            token_to_check = access_jwt or refresh_jwt
            if token_to_check:
                try:
                    # Decode without verification to see what's inside
                    payload = jwt.decode(token_to_check, options={"verify_signature": False})
                    logger.info(f"✅ Token decoded (unverified): {payload}")
                    
                    # Now try with verification
                    try:
                        verified_payload = jwt.decode(
                            token_to_check, 
                            self.jwt_secret, 
                            algorithms=["HS256"]
                        )
                        logger.info(f"✅ Token VERIFIED successfully!")
                        logger.info(f"   upid field present: {'upid' in verified_payload}")
                        logger.info(f"   uid field present: {'uid' in verified_payload}")
                    except jwt.ExpiredSignatureError:
                        logger.error("❌ Token verification FAILED: Token has expired")
                    except jwt.InvalidTokenError as e:
                        logger.error(f"❌ Token verification FAILED: {type(e).__name__}: {str(e)}")
                        logger.error("   This usually means JWT_SECRET mismatch!")
                        
                except Exception as e:
                    logger.error(f"❌ Could not decode token at all: {type(e).__name__}: {str(e)}")
            else:
                logger.warning("⚠️  NO JWT TOKEN found in request!")
            
            logger.info("=" * 80)
        
        # Continue to next middleware
        response = await call_next(request)
        
        # Log response status for auth endpoints
        if request.url.path.startswith("/api/v1/"):
            logger.info(f"📤 Response status: {response.status_code}")
        
        return response
