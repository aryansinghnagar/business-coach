"""
Deprecated: use services.azure_foundry instead.

This module is kept for backward compatibility. All functionality has moved
to Azure AI Foundry. Import from azure_foundry directly.
"""

from services.azure_foundry import (
    AzureFoundryService,
    get_foundry_service,
    get_openai_service,  # alias
)

__all__ = ["AzureFoundryService", "get_foundry_service", "get_openai_service"]
