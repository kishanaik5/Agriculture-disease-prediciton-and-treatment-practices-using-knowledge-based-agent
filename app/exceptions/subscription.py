"""
Custom exceptions for subscription service errors.
Provides clear distinction between quota limits, invalid subscriptions, and service availability.
"""

class SubscriptionError(Exception):
    """Base exception for subscription-related errors"""
    pass


class SubscriptionLimitExceeded(SubscriptionError):
    """User has exceeded their subscription quota"""
    pass


class SubscriptionInvalid(SubscriptionError):
    """Subscription ID is invalid, expired, or not found"""
    pass


class SubscriptionServiceUnavailable(SubscriptionError):
    """Subscription service is down, unreachable, or timing out"""
    pass
