from __future__ import annotations

import streamlit as st

AUTH_STATUS_KEY = "auth_status"
USERNAME_KEY = "username"
NAME_KEY = "name"


def init_auth_state() -> None:
    for key in (AUTH_STATUS_KEY, USERNAME_KEY, NAME_KEY):
        st.session_state.setdefault(key, None)


def is_authenticated() -> bool:
    return st.session_state.get(AUTH_STATUS_KEY) is True


def set_authenticated(username: str, name: str) -> None:
    st.session_state[AUTH_STATUS_KEY] = True
    st.session_state[USERNAME_KEY] = username
    st.session_state[NAME_KEY] = name


def clear_auth() -> None:
    for key in (AUTH_STATUS_KEY, USERNAME_KEY, NAME_KEY):
        st.session_state[key] = None
    for key in (
        "assessment_step",
        "assessment_data",
        "assessment_id",
        "completed_assessment_id",
        "latest_result",
        "nav_destination",
        "pending_navigation",
        "signin_username",
        "signin_password",
    ):
        st.session_state.pop(key, None)
