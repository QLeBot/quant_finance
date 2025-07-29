"""
Configuration for Powens Integration

This module contains configuration settings for the Powens integration.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class PowensConfig:
    """Configuration class for Powens integration"""
    
    # API Configuration
    API_BASE_URL = os.getenv("POWENS_API_BASE_URL", "https://api.powens.com")
    API_KEY = os.getenv("POWENS_API_KEY")
    
    # Webview Configuration
    WEBVIEW_BASE_URL = os.getenv("POWENS_WEBVIEW_BASE_URL", "https://webview.powens.com")
    REDIRECT_URL = os.getenv("POWENS_REDIRECT_URL", "http://localhost:3000/callback")
    
    # Data Configuration
    CACHE_DURATION = int(os.getenv("POWENS_CACHE_DURATION", "3600"))  # 1 hour in seconds
    MAX_RETRIES = int(os.getenv("POWENS_MAX_RETRIES", "3"))
    REQUEST_TIMEOUT = int(os.getenv("POWENS_REQUEST_TIMEOUT", "30"))
    
    # Analysis Configuration
    DEFAULT_ANALYSIS_MONTHS = int(os.getenv("POWENS_DEFAULT_ANALYSIS_MONTHS", "6"))
    MAX_TRANSACTION_HISTORY = int(os.getenv("POWENS_MAX_TRANSACTION_HISTORY", "90"))  # days
    
    # Risk Metrics Configuration
    SYNC_RISK_THRESHOLD = int(os.getenv("POWENS_SYNC_RISK_THRESHOLD", "168"))  # 7 days in hours
    CONCENTRATION_WARNING_THRESHOLD = float(os.getenv("POWENS_CONCENTRATION_WARNING_THRESHOLD", "0.5"))
    
    # Export Configuration
    DEFAULT_EXPORT_DIR = os.getenv("POWENS_EXPORT_DIR", "powens_data")
    EXPORT_FORMATS = ["csv", "json", "xlsx"]
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration and return validation results
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required environment variables
        if not cls.API_KEY:
            validation_results["valid"] = False
            validation_results["errors"].append("POWENS_API_KEY is required")
        
        if not cls.REDIRECT_URL:
            validation_results["warnings"].append("POWENS_REDIRECT_URL not set, using default")
        
        # Check configuration values
        if cls.CACHE_DURATION < 300:  # Less than 5 minutes
            validation_results["warnings"].append("Cache duration is very short")
        
        if cls.MAX_RETRIES < 1:
            validation_results["errors"].append("MAX_RETRIES must be at least 1")
            validation_results["valid"] = False
        
        if cls.REQUEST_TIMEOUT < 5:
            validation_results["warnings"].append("Request timeout is very short")
        
        return validation_results
    
    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """
        Get headers for API requests
        
        Returns:
            Dictionary with API headers
        """
        return {
            "Authorization": f"Bearer {cls.API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "QuantFinance-Powens-Integration/1.0"
        }
    
    @classmethod
    def get_webview_config(cls) -> Dict[str, Any]:
        """
        Get webview configuration
        
        Returns:
            Dictionary with webview configuration
        """
        return {
            "base_url": cls.WEBVIEW_BASE_URL,
            "redirect_url": cls.REDIRECT_URL,
            "timeout": cls.REQUEST_TIMEOUT
        } 