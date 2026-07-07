"""Authentication and security routing for Chigma."""

from __future__ import annotations

import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from src.api.auth import (
    RoleChecker,
    Roles,
    audit_col,
    create_access_token,
    get_current_user,
    hash_password,
    log_audit,
    users_col,
    verify_password,
)

router = APIRouter()

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class UserRegister(BaseModel):
    username: str = Field(..., description="Unique alphanumeric username")
    password: str = Field(..., description="Min 8 characters password")
    role: str = Field(Roles.OBSERVER, description="Observer, Operator, or Commander")
    name: str = Field(..., description="Full name / Rank of military personnel")

class UserResponse(BaseModel):
    username: str
    role: str
    name: str
    created_at: str | None = None

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> dict[str, Any]:
    """Obtain JWT Bearer access token for command access verification."""
    user = users_col.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        # Log failed attempt
        log_audit(form_data.username, "LOGIN_FAILED", "Failed authentication attempt due to invalid credentials", "WARNING")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    log_audit(user["username"], "LOGIN_SUCCESS", f"Authorized session initialized for {user['role']}", "INFO")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"]
    }

@router.post("/register", response_model=UserResponse)
async def register_user(
    new_user: UserRegister,
    current_user: dict[str, Any] = Depends(RoleChecker([Roles.COMMANDER]))
) -> dict[str, Any]:
    """Register new military/operators personnel. Restricted to Commander role."""
    if new_user.role not in [Roles.COMMANDER, Roles.OPERATOR, Roles.OBSERVER]:
        raise HTTPException(status_code=400, detail="Invalid security role specified")

    if users_col.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already registered in security database")

    hashed = hash_password(new_user.password)
    user_doc = {
        "username": new_user.username,
        "password": hashed,
        "role": new_user.role,
        "name": new_user.name,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat()
    }

    users_col.insert_one(user_doc)
    log_audit(
        current_user["username"],
        "REGISTER_USER",
        f"Registered new security clearance level {new_user.role} for {new_user.username}",
        "INFO"
    )

    return {
        "username": new_user.username,
        "role": new_user.role,
        "name": new_user.name,
        "created_at": user_doc["created_at"]
    }

@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """Get profile details of current session."""
    return {
        "username": current_user["username"],
        "role": current_user.get("role", Roles.OBSERVER),
        "name": current_user.get("name", "Observer Personnel"),
        "created_at": current_user.get("created_at")
    }

@router.get("/audit")
async def get_audit_logs(
    limit: int = 50,
    current_user: dict[str, Any] = Depends(RoleChecker([Roles.COMMANDER]))
) -> list[dict[str, Any]]:
    """Retrieve system tactical audit logs. Restricted to Commander role."""
    logs = list(audit_col.find({}).sort("timestamp", -1).limit(limit))
    formatted_logs = []
    for log in logs:
        log_copy = log.copy()
        if "_id" in log_copy:
            log_copy["_id"] = str(log_copy["_id"])
        formatted_logs.append(log_copy)
    return formatted_logs
