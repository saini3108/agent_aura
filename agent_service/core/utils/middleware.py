import time
import json
import logging
from typing import Callable, Dict, Any, Optional, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from datetime import datetime
import uuid

from agent_service.core.utils.logging_config import AuditLogger

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = AuditLogger("api_requests")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Start timing
        start_time = time.time()

        # Log request
        await self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            await self._log_response(request, response, request_id, process_time)

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log error
            await self._log_error(request, e, request_id, process_time)

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )

    async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request"""

        # Get client information
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Get request body if applicable
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await self._get_request_body(request)
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")

        # Log request
        self.audit_logger.logger.info(
            "API request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_host": client_host,
                "user_agent": user_agent,
                "body": body,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "api_request"
            }
        )

    async def _log_response(self, request: Request, response: Response,
                           request_id: str, process_time: float) -> None:
        """Log outgoing response"""

        # Get response body if possible
        response_body = None
        if hasattr(response, 'body'):
            try:
                response_body = response.body.decode('utf-8')
                # Limit response body size for logging
                if len(response_body) > 1000:
                    response_body = response_body[:1000] + "... (truncated)"
            except Exception:
                response_body = "Unable to decode response body"

        # Log response
        self.audit_logger.logger.info(
            "API response sent",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
                "response_body": response_body,
                "process_time": process_time,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "api_response"
            }
        )

    async def _log_error(self, request: Request, error: Exception,
                        request_id: str, process_time: float) -> None:
        """Log request error"""

        # Log error
        self.audit_logger.logger.error(
            "API request error",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time": process_time,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "api_error"
            }
        )

    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get request body as JSON"""

        try:
            body = await request.body()
            if body:
                return json.loads(body.decode('utf-8'))
            return None
        except Exception:
            return None

class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware for banking API"""

    def __init__(self, app: ASGIApp, allow_origins: list = None,
                 allow_methods: list = None, allow_headers: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        # Process request
        response = await call_next(request)

        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)

        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    def __init__(self, app: ASGIApp, max_requests: int = 100,
                 window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # In-memory storage (use Redis in production)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        if await self._is_rate_limited(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds",
                    "retry_after": self.window_seconds
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Window": str(self.window_seconds)
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""

        # Use IP address as client identifier
        client_host = request.client.host if request.client else "unknown"

        # You could also use API keys or user IDs if available
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key}"

        return f"ip:{client_host}"

    async def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""

        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Get or create client request history
        if client_id not in self.requests:
            self.requests[client_id] = []

        client_requests = self.requests[client_id]

        # Remove old requests outside the window
        client_requests[:] = [
            request_time for request_time in client_requests
            if request_time > window_start
        ]

        # Check if limit is exceeded
        if len(client_requests) >= self.max_requests:
            return True

        # Add current request
        client_requests.append(current_time)

        return False

    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""

        if client_id not in self.requests:
            return self.max_requests

        current_requests = len(self.requests[client_id])
        return max(0, self.max_requests - current_requests)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for banking API"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = AuditLogger("security")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Security checks
        security_issues = await self._check_security(request)

        if security_issues:
            # Log security issue
            self.audit_logger.logger.warning(
                "Security check failed",
                extra={
                    "url": str(request.url),
                    "method": request.method,
                    "client_host": request.client.host if request.client else "unknown",
                    "issues": security_issues,
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "security_check"
                }
            )

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Security check failed",
                    "issues": security_issues
                }
            )

        # Process request
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    async def _check_security(self, request: Request) -> List[str]:
        """Perform security checks"""

        issues = []

        # Check for suspicious patterns in URL
        suspicious_patterns = [
            "../", "..\\", "<script", "javascript:", "eval(",
            "exec(", "system(", "shell_exec(", "passthru("
        ]

        url_str = str(request.url).lower()
        for pattern in suspicious_patterns:
            if pattern in url_str:
                issues.append(f"Suspicious pattern detected in URL: {pattern}")

        # Check request headers
        user_agent = request.headers.get("user-agent", "").lower()
        if not user_agent or "bot" in user_agent or "crawler" in user_agent:
            # Allow legitimate monitoring/health check bots
            if request.url.path not in ["/health", "/api/v1/health"]:
                issues.append("Suspicious or missing User-Agent header")

        # Check for SQL injection patterns in query parameters
        for param_name, param_value in request.query_params.items():
            if self._check_sql_injection(param_value):
                issues.append(f"Potential SQL injection in parameter: {param_name}")

        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            issues.append("Request size exceeds maximum allowed")

        return issues

    def _check_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns"""

        sql_patterns = [
            "union", "select", "insert", "update", "delete", "drop",
            "create", "alter", "exec", "execute", "sp_", "xp_",
            "/*", "*/", "--", ";"
        ]

        value_lower = value.lower()
        for pattern in sql_patterns:
            if pattern in value_lower:
                return True

        return False

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_endpoint": {},
            "requests_by_method": {},
            "response_times": [],
            "errors_total": 0,
            "errors_by_status": {}
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()

        # Increment total requests
        self.metrics["requests_total"] += 1

        # Track by endpoint
        endpoint = request.url.path
        if endpoint not in self.metrics["requests_by_endpoint"]:
            self.metrics["requests_by_endpoint"][endpoint] = 0
        self.metrics["requests_by_endpoint"][endpoint] += 1

        # Track by method
        method = request.method
        if method not in self.metrics["requests_by_method"]:
            self.metrics["requests_by_method"][method] = 0
        self.metrics["requests_by_method"][method] += 1

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time = time.time() - start_time
        self.metrics["response_times"].append(response_time)

        # Keep only last 1000 response times
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

        # Track errors
        if response.status_code >= 400:
            self.metrics["errors_total"] += 1
            status_code = response.status_code
            if status_code not in self.metrics["errors_by_status"]:
                self.metrics["errors_by_status"][status_code] = 0
            self.metrics["errors_by_status"][status_code] += 1

        # Add metrics headers
        response.headers["X-Response-Time"] = str(response_time)

        return response

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""

        # Calculate average response time
        avg_response_time = (
            sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            if self.metrics["response_times"] else 0
        )

        # Calculate error rate
        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0 else 0
        )

        return {
            "requests_total": self.metrics["requests_total"],
            "requests_by_endpoint": self.metrics["requests_by_endpoint"],
            "requests_by_method": self.metrics["requests_by_method"],
            "errors_total": self.metrics["errors_total"],
            "errors_by_status": self.metrics["errors_by_status"],
            "error_rate": error_rate,
            "average_response_time": avg_response_time,
            "total_response_times": len(self.metrics["response_times"])
        }

# Global metrics instance
metrics_middleware = None

def get_metrics_middleware() -> MetricsMiddleware:
    """Get global metrics middleware instance"""
    global metrics_middleware
    if metrics_middleware is None:
        metrics_middleware = MetricsMiddleware(None)
    return metrics_middleware
