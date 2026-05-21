from __future__ import annotations

import html
import uuid
from typing import Any

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 120
HITL_RESUME_TIMEOUT = 120

st.set_page_config(
    page_title="MediGraph AI",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    """Initialize Streamlit session state for a HITL medical workflow."""
    defaults = {
        "thread_id": "",
        "workflow_data": {},
        "workflow_status": "idle",
        "final_report": "",
        "current_interrupt": None,
        "chat_history": [],
        "timeline": [],
        "api_health": None,
        "last_error": "",
        "dark_mode": False,
        "patient_id": "",
        "patient_name": "",
        "last_performance_ms": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_styles() -> None:
    """Apply custom CSS for a premium medical AI dashboard."""
    # The visual identity follows the provided light medical SaaS reference.
    bg = "#f6f8fb"
    panel = "#ffffff"
    text = "#172033"
    muted = "#667085"
    border = "#d7dee8"
    subtle = "#f3f6fa"
    primary = "#0f766e"
    primary_dark = "#115e59"
    accent = "#2563a8"
    navy = "#111827"

    st.markdown(
        f"""
        <style>
        :root {{
            --medical-bg: {bg};
            --medical-panel: {panel};
            --medical-text: {text};
            --medical-muted: {muted};
            --medical-border: {border};
            --medical-subtle: {subtle};
            --medical-primary: {primary};
            --medical-primary-dark: {primary_dark};
            --medical-accent: {accent};
        }}
        .stApp {{
            background:
                radial-gradient(circle at 88% 8%, rgba(56,189,248,.12), transparent 28%),
                radial-gradient(circle at 18% 0%, rgba(20,184,166,.10), transparent 32%),
                linear-gradient(180deg, {bg} 0%, #f2f7fb 100%);
            color: {text};
        }}
        .stApp, .stApp div, .stApp p, .stApp span, .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stMarkdown, .stMarkdown p {{
            color: {text};
        }}
        .block-container {{ max-width: 1540px; padding-top: 1.1rem; padding-bottom: 2.2rem; }}
        #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
        section[data-testid="stSidebar"] {{
            background: rgba(255,255,255,.78);
            backdrop-filter: blur(22px);
            border-right: 1px solid {border};
        }}
        section[data-testid="stSidebar"] * {{ color: {text} !important; }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: rgba(255,255,255,.74);
            border: 1px solid {border};
            border-radius: 12px;
            color: {text};
        }}
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: .8rem;
            margin: .35rem 0 2rem;
        }}
        .sidebar-logo-mark {{
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            color: {primary};
            font-size: 1.55rem;
            font-weight: 950;
            background: #ffffff;
            border: 2px solid {primary};
            box-shadow: 0 14px 28px rgba(15,118,110,.18);
        }}
        .sidebar-logo-title {{ font-weight: 950; font-size: 1.05rem; }}
        .sidebar-logo-subtitle {{ color: {muted} !important; font-size: .8rem; margin-top: .1rem; }}
        .sidebar-section {{
            margin: 1.4rem 0 .55rem;
            color: {muted} !important;
            font-size: .72rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .08rem;
        }}
        .nav-item {{
            display: flex;
            align-items: center;
            gap: .7rem;
            padding: .72rem .82rem;
            border-radius: 12px;
            margin-bottom: .25rem;
            color: {text} !important;
            font-weight: 850;
            font-size: .9rem;
        }}
        .nav-item.active {{
            color: {primary} !important;
            background: rgba(20,184,166,.10);
            box-shadow: inset 0 0 0 1px rgba(20,184,166,.10);
        }}
        .sidebar-help {{
            margin-top: 1.5rem;
            border: 1px solid {border};
            background: rgba(255,255,255,.62);
            border-radius: 16px;
            padding: .9rem;
            color: {muted} !important;
            font-size: .86rem;
            line-height: 1.45;
        }}
        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: .9rem;
            color: {text};
        }}
        .brand-lockup {{
            display: flex;
            align-items: center;
            gap: .75rem;
        }}
        .brand-mark {{
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, {primary} 0%, {accent} 100%);
            color: white;
            font-weight: 950;
            font-size: 1.45rem;
            box-shadow: 0 14px 28px rgba(15,118,110,.22);
        }}
        .brand-name {{
            font-weight: 950;
            font-size: 1.05rem;
            letter-spacing: .01rem;
        }}
        .brand-subtitle {{
            color: {muted};
            font-size: .86rem;
            margin-top: .1rem;
        }}
        .header-actions {{
            display: flex;
            align-items: center;
            gap: .55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}
        .hero {{
            position: relative;
            overflow: hidden;
            background: rgba(255,255,255,.0);
            border-radius: 0;
            padding: .25rem 0 1.25rem;
            margin-bottom: 1rem;
            color: {text};
            box-shadow: none;
            animation: fadeIn .35s ease;
        }}
        .hero::after {{ display: none; }}
        .hero h1 {{ margin: 0; font-size: clamp(1.55rem, 3vw, 2rem); font-weight: 950; letter-spacing: 0; color: {navy}; }}
        .hero p {{ margin: .45rem 0 0; color: {muted}; line-height: 1.55; max-width: 850px; font-weight: 650; }}
        .hero-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1rem;
            align-items: start;
        }}
        .hero-stats {{
            display: grid;
            grid-template-columns: repeat(3, minmax(105px, 1fr));
            gap: .75rem;
            min-width: 360px;
        }}
        .hero-stat {{
            border: 1px solid {border};
            background: rgba(255,255,255,.78);
            border-radius: 999px;
            padding: .7rem 1rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 12px 26px rgba(15,23,42,.06);
            color: {navy};
            text-align: center;
        }}
        .hero-stat strong {{ display: inline; font-size: .92rem; margin-right: .32rem; }}
        .hero-stat span {{ color: {navy}; font-size: .86rem; font-weight: 850; }}
        .badge {{
            display: inline-flex; align-items: center; gap: .35rem;
            border: 1px solid {border};
            background: rgba(255,255,255,.70);
            color: {primary};
            padding: .3rem .7rem; border-radius: 999px;
            font-size: .82rem; font-weight: 850; margin-bottom: .75rem;
        }}
        .medical-chip {{
            display: inline-flex;
            align-items: center;
            border: 1px solid {border};
            background: {panel};
            color: {navy};
            padding: .38rem .68rem;
            border-radius: 999px;
            font-size: .82rem;
            font-weight: 850;
            box-shadow: 0 8px 20px rgba(15,23,42,.05);
        }}
        .panel {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 1.3rem;
            box-shadow: 0 18px 45px rgba(15,23,42,.07);
            animation: fadeIn .35s ease;
        }}
        .panel-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .8rem;
            font-size: 1rem;
            font-weight: 950;
            color: {navy};
            margin-bottom: .72rem;
            padding-bottom: .65rem;
            border-bottom: 1px solid {border};
        }}
        .panel-title span {{
            color: {muted};
            font-size: .78rem;
            font-weight: 800;
        }}
        .section-kicker {{
            color: {primary};
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .05rem;
            text-transform: uppercase;
            margin-bottom: .25rem;
        }}
        .kpi-card {{
            text-align: center;
            background: linear-gradient(180deg, {panel} 0%, {subtle} 190%);
            border: 1px solid {border};
            border-radius: 16px;
            padding: 1.15rem .7rem;
            min-height: 150px;
            box-shadow: 0 10px 24px rgba(15,23,42,.055);
        }}
        .kpi-icon {{
            width: 58px;
            height: 58px;
            border-radius: 999px;
            margin: 0 auto .75rem;
            display: grid;
            place-items: center;
            font-size: 1.55rem;
            border: 1px solid {border};
            background: rgba(20,184,166,.10);
            color: {primary};
        }}
        .kpi-icon.red {{ background: rgba(239,68,68,.10); color: #dc2626; }}
        .kpi-icon.blue {{ background: rgba(59,130,246,.12); color: #2563eb; }}
        .kpi-icon.purple {{ background: rgba(124,58,237,.10); color: #6d28d9; }}
        .kpi-label {{
            color: {muted};
            font-size: .76rem;
            text-transform: uppercase;
            font-weight: 900;
            letter-spacing: .04rem;
        }}
        .kpi-value {{
            color: {navy};
            font-size: 1.55rem;
            font-weight: 950;
            margin-top: .35rem;
            word-break: break-word;
        }}
        .severity {{
            display: inline-block;
            border-radius: 999px;
            padding: .38rem .78rem;
            font-weight: 900;
            font-size: .86rem;
        }}
        .severity-faible {{ background: #dcfce7; color: #166534; }}
        .severity-modere {{ background: #fef3c7; color: #92400e; }}
        .severity-eleve {{ background: #ffedd5; color: #9a3412; }}
        .severity-critique {{ background: #fee2e2; color: #991b1b; }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .34rem .72rem;
            border-radius: 999px;
            border: 1px solid {border};
            background: {subtle};
            color: {text};
            font-size: .84rem;
            font-weight: 900;
            margin-left: .45rem;
        }}
        .clinical-strip {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .75rem;
            margin: .8rem 0 1rem;
        }}
        .clinical-strip-card {{
            border: 1px solid {border};
            background: {subtle};
            border-radius: 12px;
            padding: .78rem .9rem;
        }}
        .clinical-strip-card small {{
            display: block;
            color: {muted};
            font-weight: 850;
            text-transform: uppercase;
            font-size: .68rem;
            letter-spacing: .04rem;
            margin-bottom: .2rem;
        }}
        .clinical-strip-card strong {{
            color: {text};
            font-size: .98rem;
        }}
        .score-meter {{
            margin-top: .65rem;
            height: .58rem;
            border-radius: 999px;
            background: {subtle};
            overflow: hidden;
            border: 1px solid {border};
        }}
        .score-meter > div {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
        }}
        .score-caption {{
            display: flex;
            justify-content: space-between;
            color: {muted};
            font-size: .78rem;
            font-weight: 800;
            margin-top: .32rem;
        }}
        .timeline-item {{
            position: relative;
            border-left: 2px solid #99f6e4;
            padding: .05rem 0 .95rem 1rem;
            margin-left: .55rem;
        }}
        .timeline-item::before {{
            content: "";
            position: absolute;
            left: -7px;
            top: .22rem;
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: #0f766e;
            border: 2px solid {panel};
        }}
        .timeline-title {{ font-weight: 900; color: {text}; }}
        .timeline-subtitle {{ color: {muted}; font-size: .9rem; margin-top: .15rem; }}
        .summary-alert {{
            border: 1px solid #fecaca;
            background: #fff1f2;
            color: #991b1b;
            border-radius: 14px;
            padding: 1rem;
            margin-top: .85rem;
            font-weight: 750;
        }}
        .summary-alert strong {{ color: #b91c1c; }}
        .disclaimer-box {{
            border: 1px solid {border};
            background: rgba(255,255,255,.66);
            color: {muted};
            border-radius: 14px;
            padding: .85rem 1rem;
            margin-top: .85rem;
            font-size: .88rem;
            line-height: 1.45;
        }}
        .report-box {{
            background: {subtle};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 1.25rem;
            max-height: 760px;
            overflow: auto;
        }}
        .progress-line {{
            color: {muted};
            font-size: .9rem;
            margin-top: -.25rem;
            margin-bottom: .7rem;
        }}
        div[data-testid="stChatMessage"] {{
            border: 1px solid {border};
            border-radius: 12px;
            padding: .55rem .7rem;
            margin-bottom: .65rem;
            background: {subtle};
            box-shadow: none;
        }}
        div[data-testid="stChatMessage"] p {{
            line-height: 1.45;
        }}
        .context-box {{
            max-height: 420px;
            overflow: auto;
            border: 1px solid {border};
            border-radius: 12px;
            padding: .85rem;
            background: {subtle};
        }}
        .empty-state {{
            border: 1px dashed {border};
            border-radius: 12px;
            padding: 1rem;
            color: {muted};
            background: {subtle};
        }}
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            border-radius: 12px !important;
            border: 1px solid {primary} !important;
            background: linear-gradient(135deg, {primary} 0%, {accent} 100%) !important;
            color: white !important;
            font-weight: 900 !important;
            min-height: 2.65rem;
            box-shadow: 0 10px 22px rgba(15,118,110,.18);
        }}
        .stButton > button *, .stDownloadButton > button *, .stFormSubmitButton > button * {{
            color: white !important;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: rgba(255,255,255,.74) !important;
            border: 1px solid {border} !important;
            box-shadow: 0 8px 18px rgba(15,23,42,.05) !important;
            color: {text} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button * {{
            color: {text} !important;
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
            border-radius: 12px !important;
            color: {text} !important;
            background: #ffffff !important;
            border-color: {border} !important;
        }}
        button[data-baseweb="tab"] {{
            font-weight: 850;
            border-radius: 10px 10px 0 0;
        }}
        div[data-testid="stProgress"] > div > div > div {{
            background: linear-gradient(90deg, {primary} 0%, {accent} 100%);
        }}
        @media (max-width: 1050px) {{
            .hero-grid {{ grid-template-columns: 1fr; }}
            .hero-stats {{ min-width: 0; grid-template-columns: repeat(3, 1fr); }}
            .clinical-strip {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 680px) {{
            .topbar {{ align-items: flex-start; flex-direction: column; }}
            .header-actions {{ justify-content: flex-start; }}
            .hero {{ padding: 1.1rem; }}
            .hero-stats {{ grid-template-columns: 1fr; }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(7px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Professional clinical dashboard overrides */
        .stApp {{
            background: linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
        }}
        .block-container {{
            max-width: 1480px;
            padding: 1.25rem 2rem 2.5rem;
        }}
        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid #d9e1ea;
            box-shadow: 10px 0 30px rgba(15, 23, 42, .035);
        }}
        .sidebar-logo {{
            margin: .25rem 0 1.55rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e4eaf1;
        }}
        .sidebar-logo-mark {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            color: #ffffff;
            background: #0f766e;
            border: 0;
            box-shadow: none;
        }}
        .nav-item {{
            border-radius: 8px;
            padding: .66rem .72rem;
            font-size: .88rem;
            color: #344054 !important;
        }}
        .nav-item.active {{
            background: #eaf6f4;
            color: #0f766e !important;
            box-shadow: inset 3px 0 0 #0f766e;
        }}
        .sidebar-help {{
            border-radius: 8px;
            background: #f8fafc;
            border: 1px solid #e4eaf1;
            box-shadow: none;
        }}
        .hero {{
            background: #ffffff;
            border: 1px solid #d9e1ea;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .055);
        }}
        .hero h1 {{
            font-size: clamp(1.45rem, 2.2vw, 2rem);
            letter-spacing: 0;
        }}
        .hero p {{
            max-width: 760px;
            font-weight: 500;
        }}
        .hero-stats {{
            grid-template-columns: repeat(3, minmax(92px, 1fr));
            min-width: 320px;
        }}
        .hero-stat {{
            border-radius: 8px;
            background: #f8fafc;
            padding: .68rem .78rem;
            box-shadow: none;
        }}
        .panel {{
            border-radius: 10px;
            border: 1px solid #d9e1ea;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .055);
            padding: 1.05rem;
        }}
        .panel-title {{
            font-size: .98rem;
            margin-bottom: .9rem;
            padding-bottom: .72rem;
        }}
        .section-kicker {{
            color: #0f766e;
            letter-spacing: .06rem;
        }}
        .clinical-strip {{
            grid-template-columns: 1.1fr 1fr .9fr;
            gap: .65rem;
        }}
        .clinical-strip-card {{
            border-radius: 8px;
            background: #f8fafc;
            padding: .72rem .78rem;
        }}
        .kpi-card {{
            text-align: left;
            min-height: 112px;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: none;
            padding: .9rem;
        }}
        .kpi-icon {{
            width: 34px;
            height: 34px;
            margin: 0 0 .7rem;
            border-radius: 8px;
            font-size: .82rem;
            font-weight: 950;
            background: #edf7f5;
            border: 1px solid #cde7e2;
        }}
        .kpi-value {{
            font-size: 1.18rem;
            line-height: 1.2;
        }}
        .kpi-label {{
            margin-top: .22rem;
            font-size: .69rem;
        }}
        .severity, .status-badge {{
            border-radius: 8px;
        }}
        .score-meter {{
            height: .46rem;
            background: #eef2f6;
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #d9e1ea;
            box-shadow: 0 6px 16px rgba(15, 23, 42, .04);
        }}
        .timeline-item {{
            border-left-color: #b8d8d4;
        }}
        .timeline-item::before {{
            background: #0f766e;
        }}
        .empty-state, .context-box, .report-box, .summary-alert, .disclaimer-box {{
            border-radius: 8px;
        }}
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            border-radius: 8px !important;
            background: #0f766e !important;
            border: 1px solid #0f766e !important;
            box-shadow: none !important;
            min-height: 2.55rem;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
            background: #115e59 !important;
            border-color: #115e59 !important;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            border-radius: 8px !important;
            background: #ffffff !important;
        }}
        .stTextInput input, .stTextArea textarea {{
            border-radius: 8px !important;
            border: 1px solid #cfd8e3 !important;
            box-shadow: none !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: #0f766e !important;
        }}
        button[data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding-top: .55rem;
            padding-bottom: .55rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _extract_error_detail(response: requests.Response) -> str:
    """Extract a human-readable FastAPI error."""
    try:
        payload = response.json()
    except ValueError:
        return f"Erreur API {response.status_code}: {response.text}"
    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("error") or str(detail)
        steps = detail.get("next_steps")
        return f"{message}\n\nÉtapes suggérées: {' | '.join(steps)}" if steps else message
    return str(detail)


def api_get(path: str, timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
    """Call a FastAPI GET endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("La requête a expiré.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("API indisponible. Lance: uvicorn backend.app.api:app --reload") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(_extract_error_detail(response)) from exc


def api_post(path: str, payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
    """Call a FastAPI POST endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("La requête a expiré. Le workflow est peut-être en attente.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("API indisponible. Lance: uvicorn backend.app.api:app --reload") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(_extract_error_detail(response)) from exc


def api_post_pdf(payload: dict[str, Any]) -> bytes:
    """Export a Markdown report as PDF through FastAPI."""
    try:
        response = requests.post(f"{API_BASE_URL}/export/pdf", json=payload, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as exc:
        detail = _extract_error_detail(response) if "response" in locals() else str(exc)
        raise RuntimeError(detail) from exc


def severity_class(severity: str | None) -> str:
    """Map a severity label to a CSS badge class."""
    normalized = (severity or "faible").lower().replace("é", "e")
    if "critique" in normalized:
        return "severity-critique"
    if "eleve" in normalized:
        return "severity-eleve"
    if "modere" in normalized:
        return "severity-modere"
    return "severity-faible"


def status_label(status: str | None) -> str:
    labels = {
        "idle": "En attente",
        "starting": "Demarrage",
        "running": "Analyse",
        "waiting_patient_response": "Question patient",
        "waiting_physician_review": "Supervision humaine",
        "finished": "Termine",
    }
    return labels.get(status or "idle", status or "En attente")


def append_chat(role: str, content: str) -> None:
    """Append one message to the visible conversation."""
    if not content:
        return
    message = {"role": role, "content": content}
    if not st.session_state.chat_history or st.session_state.chat_history[-1] != message:
        st.session_state.chat_history.append(message)


def add_timeline(title: str, subtitle: str) -> None:
    """Append a unique workflow event to the timeline."""
    event = {"title": title, "subtitle": subtitle}
    if event not in st.session_state.timeline:
        st.session_state.timeline.append(event)


def extract_interrupt(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract LangGraph interrupt payload from API JSON."""
    interrupt_data = data.get("__interrupt__") if isinstance(data, dict) else None
    if not interrupt_data:
        return None
    first = interrupt_data[0] if isinstance(interrupt_data, list) else interrupt_data
    if isinstance(first, dict):
        value = first.get("value", first)
        return value if isinstance(value, dict) else {"type": "unknown", "message": str(value)}
    return {"type": "unknown", "message": str(first)}


def result_interrupt(result: dict[str, Any], data: dict[str, Any]) -> dict[str, Any] | None:
    """Read interrupt payload from normalized or legacy API responses."""
    interrupt_payload = result.get("interrupt")
    if isinstance(interrupt_payload, dict):
        return interrupt_payload
    return extract_interrupt(data)


def sync_from_result(result: dict[str, Any]) -> None:
    """Synchronize frontend state with a LangGraph API response."""
    data = result.get("data") or result.get("workflow_state") or {}
    status = result.get("status", "running")
    st.session_state.thread_id = result.get("thread_id", st.session_state.thread_id)
    st.session_state.workflow_data = data
    interrupt_payload = result_interrupt(result, data)
    if isinstance(interrupt_payload, dict):
        for key in ("clinical_score", "severity_level", "clinical_category"):
            if interrupt_payload.get(key) is not None:
                data[key] = interrupt_payload.get(key)
        st.session_state.workflow_data = data
    if status in {"ok", "running"} and interrupt_payload:
        interrupt_type = interrupt_payload.get("type")
        if interrupt_type == "patient_question":
            status = "waiting_patient_response"
        elif interrupt_type == "physician_review":
            status = "waiting_physician_review"
    st.session_state.workflow_status = status
    st.session_state.last_performance_ms = result.get("performance_ms", st.session_state.last_performance_ms)
    if status == "finished" and data.get("final_report"):
        st.session_state.final_report = data.get("final_report")
    st.session_state.current_interrupt = interrupt_payload

    if status == "waiting_patient_response" and interrupt_payload:
        index = interrupt_payload.get("question_index", "?")
        question = interrupt_payload.get("question") or interrupt_payload.get("next_question", "")
        append_chat("assistant", f"**Question {index}/5**\n\n{question}")
        perf = result.get("performance_ms")
        suffix = f" ({perf:.0f} ms)" if isinstance(perf, (int, float)) else ""
        add_timeline("Question agent", f"Question {index}/5 en attente de reponse{suffix}")
    elif status == "waiting_physician_review" and interrupt_payload:
        append_chat("assistant", "Les 5 reponses sont collectees. Supervision humaine requise avant le rapport final.")
        add_timeline("Supervision humaine", "Validation supervisee en attente")
    elif status == "finished" and data.get("final_report"):
        append_chat("assistant", "Consultation terminee. Le rapport final est disponible.")
        add_timeline("Rapport final", "Rapport genere")


def check_health() -> None:
    """Check FastAPI health."""
    try:
        st.session_state.api_health = api_get("/health", timeout=8)
        st.session_state.last_error = ""
    except RuntimeError as exc:
        st.session_state.api_health = None
        st.session_state.last_error = str(exc)


def start_consultation(patient_id: str, patient_name: str, patient_case: str) -> None:
    """Start the HITL consultation workflow."""
    thread_id = f"{patient_id or 'patient'}-{uuid.uuid4().hex[:8]}"
    payload = {
        "patient_id": patient_id.strip() or None,
        "patient_name": patient_name.strip() or None,
        "session_id": thread_id,
        "patient_case": patient_case.strip(),
    }
    result = api_post("/consultation/start", payload)

    st.session_state.thread_id = result.get("thread_id", thread_id)
    st.session_state.patient_id = patient_id
    st.session_state.patient_name = patient_name
    st.session_state.chat_history = []
    st.session_state.timeline = []
    st.session_state.final_report = ""
    st.session_state.workflow_data = {}
    st.session_state.workflow_status = "starting"
    st.session_state.current_interrupt = None

    append_chat("user", f"Cas initial\n\n{patient_case.strip()}")
    add_timeline("Consultation demarree", f"Thread {st.session_state.thread_id}")
    sync_from_result(result)


def resume_consultation(resume_payload: Any) -> None:
    """Resume a LangGraph interrupt with patient answer or physician decision."""
    if not st.session_state.thread_id:
        raise RuntimeError("Aucune consultation active.")
    result = api_post(
        "/consultation/resume",
        {"thread_id": st.session_state.thread_id, "resume": resume_payload},
        timeout=HITL_RESUME_TIMEOUT,
    )
    sync_from_result(result)


def refresh_state() -> None:
    """Refresh current workflow state."""
    if not st.session_state.thread_id:
        raise RuntimeError("Aucune consultation active.")
    result = api_get(f"/consultation/{st.session_state.thread_id}")
    sync_from_result(result)


def load_report() -> None:
    """Load final report from the backend."""
    if not st.session_state.thread_id:
        raise RuntimeError("Aucune consultation active.")
    result = api_get(f"/consultation/{st.session_state.thread_id}/report")
    st.session_state.final_report = result.get("final_report", "")
    add_timeline("Rapport charge", "Rapport final recupere")


def render_sidebar() -> None:
    """Render sidebar controls and backend status."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-mark">M</div>
                <div>
                    <div class="sidebar-logo-title">MediGraph AI</div>
                    <div class="sidebar-logo-subtitle">Workflow clinique supervise</div>
                </div>
            </div>
            <div class="sidebar-section">General</div>
            <div class="nav-item active">Tableau de bord</div>
            <div class="nav-item">Consultation active</div>
            <div class="nav-item">Historique</div>
            <div class="nav-item">Rapports</div>
            <div class="nav-item">Parametres</div>
            <div class="sidebar-section">Outils</div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.dark_mode = st.toggle("Mode sombre", value=st.session_state.dark_mode)

        if st.button("Verifier API", use_container_width=True):
            check_health()

        health = st.session_state.api_health
        if health:
            st.success("Plateforme connectee")
            st.write(f"Graph: `{health.get('graph_id', '-')}`")
            st.write(f"Checkpointer: `{health.get('checkpointer', '-')}`")
            st.write(f"Simulation: `{health.get('simulation_mode', '-')}`")
        elif st.session_state.last_error:
            st.error(st.session_state.last_error)
        else:
            st.info("Verifie l'API avant de lancer une consultation.")

        st.divider()
        st.markdown("### Consultation active")
        st.code(st.session_state.thread_id or "Aucune", language="text")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Etat", use_container_width=True):
                try:
                    refresh_state()
                    st.toast("Etat synchronisé")
                except RuntimeError as exc:
                    st.error(str(exc))
        with c2:
            if st.button("Rapport", use_container_width=True):
                try:
                    load_report()
                    st.toast("Rapport charge")
                except RuntimeError as exc:
                    st.error(str(exc))

        st.markdown(
            """
            <div class="nav-item">Documentation</div>
            <div class="nav-item">Aide</div>
            <div class="sidebar-help">
                Prototype academique de workflow HITL. Les resultats servent a l'orientation et ne remplacent pas une evaluation clinique reelle.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header() -> None:
    """Render app header."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <h1>Consultation clinique multi-agents</h1>
                    <p>Workflow conversationnel HITL avec orientation clinique, supervision humaine et rapport final.</p>
                </div>
                <div class="hero-stats">
                    <div class="hero-stat"><strong>API</strong><span>FastAPI</span></div>
                    <div class="hero-stat"><strong>LG</strong><span>LangGraph</span></div>
                    <div class="hero-stat"><strong>AI</strong><span>MCP + RAG</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_form() -> None:
    """Render patient intake form."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Dossier patient <span>Admission clinique</span></div>',
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Identifiant patient", value=st.session_state.patient_id)
        with c2:
            patient_name = st.text_input("Nom du patient", value=st.session_state.patient_name)
        patient_case = st.text_area(
            "Cas clinique initial",
            value="",
            height=135,
            placeholder="Decris le motif de consultation, les symptomes et le contexte connu...",
        )
        submitted = st.form_submit_button("Demarrer la consultation", use_container_width=True)

    if submitted:
        if not patient_case.strip():
            st.error("Le cas clinique est obligatoire.")
        else:
            with st.spinner("Demarrage du workflow LangGraph..."):
                try:
                    start_consultation(patient_id, patient_name, patient_case)
                    st.success("Consultation demarree. Reponds a la premiere question.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

    st.markdown("</div>", unsafe_allow_html=True)


def render_chat() -> None:
    """Render ChatGPT-like medical conversation."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Conversation clinique <span>Echange patient</span></div>',
        unsafe_allow_html=True,
    )

    data = st.session_state.workflow_data or {}
    progress = int(data.get("question_count") or 0)
    st.progress(min(progress, 5) / 5, text=f"Progression: {min(progress, 5)}/5")
    if isinstance(st.session_state.last_performance_ms, (int, float)):
        st.caption(f"Dernier tour HITL traite en {st.session_state.last_performance_ms:.0f} ms")

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">Demarre une consultation pour recevoir la premiere question contextualisee.</div>',
            unsafe_allow_html=True,
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    interrupt_payload = st.session_state.current_interrupt
    if interrupt_payload and interrupt_payload.get("type") == "patient_question":
        with st.form("patient_answer_form", clear_on_submit=True):
            answer = st.text_area("Reponse patient", height=90, placeholder="Saisis la reponse du patient...")
            submitted = st.form_submit_button("Envoyer réponse", use_container_width=True)
        if submitted:
            if not answer.strip():
                st.error("La reponse ne peut pas etre vide.")
            else:
                append_chat("user", answer.strip())
                add_timeline("Reponse patient", answer.strip()[:100])
                try:
                    with st.status("Analyse rapide de la reponse...", expanded=True) as status:
                        status.write("Mise a jour de la memoire conversationnelle")
                        status.write("Generation de la prochaine question")
                        resume_consultation(answer.strip())
                        status.update(label="Question suivante prete", state="complete")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    elif interrupt_payload and interrupt_payload.get("type") == "physician_review":
        render_physician_review(interrupt_payload)
    elif st.session_state.final_report:
        st.success("Consultation terminee. Rapport final disponible.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_physician_review(interrupt_payload: dict[str, Any]) -> None:
    """Render the human supervision HITL review block."""
    st.markdown('<div class="section-kicker">Supervision humaine</div>', unsafe_allow_html=True)
    st.markdown("#### Validation clinique supervisee")
    st.info("Cette etape represente une validation humaine simulee avant generation du rapport final.")

    with st.expander("Synthese et recommandation", expanded=False):
        st.markdown(interrupt_payload.get("diagnostic_summary", "_Synthese indisponible._"))
        st.markdown(interrupt_payload.get("interim_care", "_Recommandation indisponible._"))

    data = st.session_state.workflow_data or {}
    with st.expander("Contexte MCP et RAG", expanded=False):
        st.markdown("**MCP**")
        st.markdown(data.get("mcp_context") or "_Donnees MCP non encore disponibles._")
        st.markdown("**RAG**")
        st.markdown(data.get("rag_context") or "_Contexte RAG non encore disponible._")

    c1, c2 = st.columns(2)
    with c1:
        validate_clicked = st.button("Valider le rapport", use_container_width=True, type="primary")
    with c2:
        revise_clicked = st.button("Demander une revision", use_container_width=True)

    if validate_clicked or revise_clicked:
        if validate_clicked:
            payload = {
                "decision": "approve",
                "treatment": "Validation humaine simulee pour demonstration academique.",
                "notes": "Rapport valide dans le cadre du workflow HITL de demonstration.",
            }
            event_label = "Rapport valide"
        else:
            payload = {
                "decision": "modify",
                "treatment": "Revision humaine simulee demandee avant interpretation finale.",
                "notes": "Une revision du contenu est demandee dans le cadre de la demonstration.",
            }
            event_label = "Revision demandee"

        append_chat("assistant", f"Supervision humaine: {event_label}.")
        add_timeline("Supervision humaine", event_label)
        with st.spinner("Generation du rapport final..."):
            try:
                resume_consultation(payload)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))


def render_kpis(data: dict[str, Any]) -> None:
    """Render animated KPI cards."""
    score = data.get("clinical_score", "-")
    severity = data.get("severity_level", "faible")
    category = data.get("clinical_category", "-")
    question_count = data.get("question_count", 0)
    try:
        score_value = int(score)
    except (TypeError, ValueError):
        score_value = 0
    meter_width = max(0, min(score_value, 15)) / 15 * 100

    cols = st.columns(4)
    metrics = [
        ("SC", "Score clinique", score, ""),
        ("GR", "Gravite", severity, "red" if "critique" in str(severity).lower() else ""),
        ("OR", "Categorie", category, "blue"),
        ("Q", "Questions HITL", f"{min(int(question_count or 0), 5)}/5", "purple"),
    ]
    for col, (icon, label, value, tone) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon {tone}">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <span class="severity {severity_class(str(severity))}">Niveau: {severity}</span>
        <span class="status-badge">Statut: {status_label(st.session_state.workflow_status)}</span>
        <div class="score-meter"><div style="width: {meter_width:.0f}%"></div></div>
        <div class="score-caption"><span>Faible</span><span>Modere</span><span>Eleve</span><span>Critique</span></div>
        """,
        unsafe_allow_html=True,
    )


def render_clinical_strip(data: dict[str, Any]) -> None:
    """Render a compact clinical summary strip above detailed tabs."""
    patient_name = st.session_state.get("patient_name") or "Patient"
    patient_id = st.session_state.get("patient_id") or "-"
    category = data.get("clinical_category") or "En cours"
    status = status_label(st.session_state.workflow_status)
    st.markdown(
        f"""
        <div class="clinical-strip">
            <div class="clinical-strip-card">
                <small>Patient</small>
                <strong>{html.escape(str(patient_name))} · {html.escape(str(patient_id))}</strong>
            </div>
            <div class="clinical-strip-card">
                <small>Orientation clinique</small>
                <strong>{html.escape(str(category))}</strong>
            </div>
            <div class="clinical-strip-card">
                <small>Workflow</small>
                <strong>{html.escape(str(status))}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    """Render final or in-progress clinical dashboard."""
    data = st.session_state.workflow_data or {}
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Tableau clinique <span>Score, gravité et contexte</span></div>',
        unsafe_allow_html=True,
    )

    if not data:
        st.markdown(
            '<div class="empty-state">Les métriques cliniques apparaîtront après les premières réponses patient.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    render_clinical_strip(data)
    render_kpis(data)
    st.divider()
    tabs = st.tabs(["Synthèse clinique", "Recommandations", "MCP", "RAG", "Workflow"])
    with tabs[0]:
        st.markdown(data.get("diagnostic_summary") or "_Synthèse non disponible tant que les 5 réponses ne sont pas collectées._")
    with tabs[1]:
        st.markdown(data.get("interim_care") or "_Recommandation non disponible._")
    with tabs[2]:
        st.markdown('<div class="context-box">', unsafe_allow_html=True)
        st.markdown(data.get("mcp_context") or "_Données MCP non encore disponibles._")
        st.markdown("</div>", unsafe_allow_html=True)
    with tabs[3]:
        st.markdown('<div class="context-box">', unsafe_allow_html=True)
        st.markdown(data.get("rag_context") or "_Contexte RAG non encore récupéré._")
        st.markdown("</div>", unsafe_allow_html=True)
    with tabs[4]:
        st.json(
            {
                "thread_id": st.session_state.thread_id,
                "next": data.get("next"),
                "status": st.session_state.workflow_status,
                "consultation_finished": data.get("consultation_finished"),
                "clinical_category": data.get("clinical_category"),
                "severity_level": data.get("severity_level"),
                "interrupt": st.session_state.current_interrupt,
            }
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline() -> None:
    """Render consultation timeline."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Timeline consultation <span>Parcours HITL</span></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.timeline:
        st.markdown(
            '<div class="empty-state">Aucun événement pour le moment.</div>',
            unsafe_allow_html=True,
        )
    else:
        for event in st.session_state.timeline[-10:]:
            title = html.escape(str(event["title"]))
            subtitle = html.escape(str(event["subtitle"]))
            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="timeline-title">{title}</div>
                    <div class="timeline-subtitle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_report() -> None:
    """Render the final report and PDF export."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Synthèse clinique <span>Rapport et recommandations</span></div>',
        unsafe_allow_html=True,
    )

    report = st.session_state.final_report
    if not report:
        data = st.session_state.workflow_data or {}
        summary = data.get("diagnostic_summary")
        interim = data.get("interim_care")
        severity = str(data.get("severity_level") or "").lower()
        if summary or interim:
            if summary:
                st.markdown(summary)
            if interim:
                st.markdown(
                    f"""
                    <div class="summary-alert">
                        <strong>Recommandation :</strong><br>{html.escape(str(interim))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if "critique" in severity:
                st.markdown(
                    '<div class="summary-alert"><strong>Alerte critique :</strong> une évaluation médicale urgente peut être nécessaire selon le contexte clinique.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="disclaimer-box">Les données MCP/RAG servent uniquement d’aide contextuelle et ne constituent pas un diagnostic définitif.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">La synthèse clinique apparaîtra après les réponses patient et la supervision humaine.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(report)
    st.markdown("</div>", unsafe_allow_html=True)

    severity = (st.session_state.workflow_data or {}).get("severity_level", "Non renseigné")
    try:
        pdf = api_post_pdf({"final_report": report, "severity_level": severity})
        st.download_button(
            "Télécharger le rapport PDF",
            data=pdf,
            file_name="rapport-medical.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except RuntimeError as exc:
        st.error(f"Export PDF indisponible: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit HITL medical app."""
    init_state()
    inject_styles()
    render_sidebar()
    render_header()

    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        render_patient_form()
    with right:
        render_dashboard()

    st.write("")
    lower_left, lower_right = st.columns([0.42, 0.58], gap="large")
    with lower_left:
        render_timeline()
        st.write("")
        render_chat()
    with lower_right:
        render_report()


if __name__ == "__main__":
    main()



