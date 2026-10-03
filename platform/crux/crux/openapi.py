"""Static OpenAPI for the agent port. Admin work stays in the Utopia UI."""

from __future__ import annotations

from typing import Any


def openapi_document() -> dict[str, Any]:
    # Agent port only. Do not add enroll/revoke/audit — those stay operator/admin.
    identity = {
        "type": "object",
        "required": ["owner_id", "agent_id", "credential_id"],
        "properties": {
            "owner_id": {"type": "string"},
            "agent_id": {"type": "string"},
            "credential_id": {"type": "string"},
        },
    }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Crux",
            "version": "0.1.0",
            "description": (
                "Agent port for Crux — Enterprise Ontology (Utopia) plus "
                "Crux — Token Optimization (Headroom). "
                "Call POST /v1/ask. remember proposes facts; it does not commit a live graph edge."
            ),
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Dependency reachability",
                    "responses": {"200": {"description": "ok"}, "503": {"description": "degraded"}},
                }
            },
            "/v1/tools": {
                "get": {"summary": "Pinned MCP tools", "responses": {"200": {"description": "manifest"}}}
            },
            "/v1/readiness": {
                "get": {
                    "summary": "Restricted-base readiness and extraction blockers",
                    "description": "Operator view. Graph extract is configured in Enterprise Ontology, never via Headroom.",
                    "responses": {"200": {"description": "operator view"}},
                }
            },
            "/v1/openapi.json": {
                "get": {"summary": "This document", "responses": {"200": {"description": "OpenAPI"}}}
            },
            "/v1/catalog": {
                "post": {
                    "summary": "Granted bases and tools for an identity",
                    "requestBody": {"content": {"application/json": {"schema": identity}}},
                    "responses": {"200": {"description": "catalog"}},
                }
            },
            "/v1/session": {
                "post": {
                    "summary": "Start a session subset of granted bases",
                    "responses": {"200": {"description": "session"}},
                }
            },
            "/v1/ask": {
                "post": {
                    "summary": "Authorized Utopia MCP call, then Headroom compress",
                    "description": "Use tool remember to propose. find_entities then entity_facts for the graph. Facts stay in Review until a human approves.",
                    "responses": {"200": {"description": "utopia + compressed + text"}},
                }
            },
        },
    }
