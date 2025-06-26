"""
ValiCred-AI: Credit Risk Model Validation System
===============================================

Production-ready application with dynamic configuration, real-time communication,
and comprehensive data upload capabilities.
"""

import streamlit as st
from src.core.clean_config_system import get_config
from src.core.app_factory import initialize_application, initialize_session_state, configure_page
from src.ui.dashboard import show_dashboard, show_system_status
from src.ui.enhanced_workflow_demo import show_enhanced_workflow_demo
from src.ui.reports_interface import show_reports, show_audit_trail
from src.ui.configuration_panel import show_configuration
from src.ui.data_upload import show_data_upload_interface

def main():
    """Main application entry point"""

    # Configure page
    configure_page()

    # Initialize session state
    initialize_session_state()

    # Initialize application components
    if not st.session_state.app_initialized:
        with st.spinner("Initializing ValiCred-AI system..."):
            mcp_engine, audit_logger, sample_loader, use_advanced = initialize_application()

            # Store in session state
            st.session_state.mcp_engine = mcp_engine
            st.session_state.audit_logger = audit_logger
            st.session_state.sample_loader = sample_loader
            st.session_state.use_advanced_config = use_advanced
            st.session_state.app_initialized = True

        if use_advanced:
            st.success("ValiCred-AI initialized with full MCP + LangGraph capabilities")
        else:
            st.info("ValiCred-AI initialized in fallback mode")

    # Get components from session state
    mcp_engine = st.session_state.mcp_engine
    audit_logger = st.session_state.audit_logger
    sample_loader = st.session_state.sample_loader

    # Sidebar navigation
    st.sidebar.title("🏦 ValiCred-AI")
    st.sidebar.markdown("**Credit Risk Model Validation**")

    # Load dynamic configuration
    config = get_config()
    ui_settings = config.get_ui_settings()

    page = st.sidebar.selectbox(
        "Navigate",
        ["Dashboard", "Data Upload", "Workflow Engine", "Reports", "Audit Trail", "Configuration", "System Status"]
    )

    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard(mcp_engine, audit_logger, sample_loader)

    elif page == "Data Upload":
        data_config = config.get_data_config()
        uploaded_data, metadata = show_data_upload_interface(data_config)

        if uploaded_data is not None:
            st.session_state.validation_data = uploaded_data
            st.session_state.data_metadata = metadata
            if audit_logger:
                audit_logger.log_data_operation("data_uploaded", metadata)
            st.success("Data uploaded successfully! You can now proceed to MCP Workflow.")

    elif page == "Workflow Engine":
        show_enhanced_workflow_demo()

    elif page == "Reports":
        show_reports(mcp_engine, audit_logger)

    elif page == "Audit Trail":
        show_audit_trail(audit_logger)

    elif page == "Configuration":
        show_configuration()

    elif page == "System Status":
        show_system_status(mcp_engine)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ValiCred-AI v1.0**")
    st.sidebar.markdown("Agent Aura Architecture")

if __name__ == "__main__":
    main()