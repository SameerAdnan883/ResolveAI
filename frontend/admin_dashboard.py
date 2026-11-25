import streamlit as st
import pandas as pd
from frontend.api_client import ResolveAIClient

def render_admin_dashboard(client: ResolveAIClient):
    st.markdown("## 🛡️ ResolveAI Supervisor & Admin Operations")
    st.caption("Oversight, deterministic business approvals, agent audit telemetry, and service analytics.")

    if "admin_token" not in st.session_state:
        st.session_state.admin_token = None
    if "admin_user" not in st.session_state:
        st.session_state.admin_user = None

    if not st.session_state.admin_token:
        st.markdown("### Admin Sign In")
        st.info("Default Admin: `admin@resolveai.com` / Password: `admin123`")
        col_em, col_pw = st.columns(2)
        with col_em:
            admin_email = st.text_input("Admin Email", value="admin@resolveai.com", key="adm_login_email")
        with col_pw:
            admin_pass = st.text_input("Password", value="admin123", type="password", key="adm_login_pass")

        if st.button("Sign In to Admin Operations", type="primary", key="adm_login_btn"):
            res = client.login(admin_email, admin_pass)
            if "access_token" in res:
                if res["user"].get("role") != "admin":
                    st.error("Access denied: You must have an administrative role.")
                else:
                    st.session_state.admin_token = res["access_token"]
                    st.session_state.admin_user = res["user"]
                    st.success("Admin authorized.")
                    st.rerun()
            else:
                st.error(res.get("error", "Authentication failed."))
        return

    token = st.session_state.admin_token
    user = st.session_state.admin_user

    # Header bar
    col_adm_info, col_adm_logout = st.columns([4, 1])
    with col_adm_info:
        st.markdown(f"**Administrator:** {user['full_name']} | Role: `Supervisor / Admin`")
    with col_adm_logout:
        if st.button("Sign Out Admin", key="adm_logout"):
            st.session_state.admin_token = None
            st.session_state.admin_user = None
            st.rerun()

    st.divider()

    # KPI Top Bar
    analytics = client.get_analytics_summary(token)
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Tickets", analytics.get("total_tickets", 0))
    kpi2.metric("Pending Approvals", analytics.get("pending_approvals", 0), delta_color="inverse")
    kpi3.metric("Resolved", analytics.get("resolved_tickets", 0))
    kpi4.metric("Escalated", analytics.get("escalated_tickets", 0))
    kpi5.metric("Avg Latency", f"{analytics.get('average_agent_latency_ms', 0)} ms")

    st.divider()

    tab_approvals, tab_all, tab_deep_dive, tab_metrics = st.tabs([
        "⚠️ Pending Human Approvals",
        "📂 All Support Tickets",
        "🔬 Agent Telemetry & Logs",
        "📊 Analytics & KPIs"
    ])

    with tab_approvals:
        st.markdown("### Human-In-The-Loop Approval Queue")
        st.caption(
            "Strict business rule: No AI agent can disburse refunds or override high-value thresholds ($150+) without supervisor authorization."
        )

        pending = client.get_pending_approvals(token)
        if not pending:
            st.success("🎉 No pending approvals! All tickets are fully compliant.")
        else:
            for item in pending:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 2, 2])
                    with c1:
                        st.markdown(f"#### 🎫 {item['ticket_id']} — {item['ticket_title']}")
                        st.write(f"**Customer Issue:** {item['complaint_text']}")
                        st.error(f"**Escalation Reason:** {item['escalation_reason']}")
                    with c2:
                        st.write(f"**Action Requested:** `{item['action_type'].upper()}`")
                        st.write(f"**Order ID:** `{item['order_id']}`")
                        st.write(f"**Amount Involved:** **${item['proposed_amount']:.2f}**")
                    with c3:
                        admin_notes = st.text_input(
                            "Supervisor Notes:",
                            key=f"notes_{item['approval_id']}",
                            placeholder="Reason for decision"
                        )
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            if st.button("✅ Approve", key=f"app_{item['approval_id']}", type="primary"):
                                res = client.review_approval(token, item["approval_id"], True, admin_notes)
                                if "error" in res:
                                    st.error(res["error"])
                                else:
                                    st.success("Action approved!")
                                    st.rerun()
                        with b_col2:
                            if st.button("❌ Reject", key=f"rej_{item['approval_id']}"):
                                res = client.review_approval(token, item["approval_id"], False, admin_notes)
                                if "error" in res:
                                    st.error(res["error"])
                                else:
                                    st.warning("Action rejected.")
                                    st.rerun()

    with tab_all:
        st.markdown("### All Support Tickets")
        filter_status = st.selectbox(
            "Filter by Status:",
            ["All", "open", "investigating", "pending_approval", "resolved", "closed"],
            key="ticket_status_filter"
        )
        selected_filter = None if filter_status == "All" else filter_status
        all_tickets = client.get_all_tickets(token, selected_filter)

        if all_tickets:
            df = pd.DataFrame(all_tickets)
            st.dataframe(
                df[["id", "customer_email", "order_id", "category", "priority", "status", "created_at"]],
                use_container_width=True
            )
        else:
            st.info("No tickets match the selected criteria.")

    with tab_deep_dive:
        st.markdown("### Detailed Ticket Investigation & Agent Execution Telemetry")
        t_id_input = st.text_input("Enter Ticket ID to inspect:", value="TCK-5001", key="inspect_ticket_input")

        if st.button("Inspect Ticket Logs", key="inspect_btn"):
            t_details = client.get_ticket_details(token, t_id_input)
            if "error" in t_details:
                st.error(t_details["error"])
            else:
                st.markdown(f"### Ticket: `{t_details['id']}` - {t_details['title']}")
                st.write(f"**Customer:** ID {t_details['customer_id']} | **Order:** {t_details['order_id']} | **Status:** `{t_details['status']}`")
                st.write(f"**Complaint:** {t_details['complaint_text']}")

                st.markdown("#### Agent Telemetry Chronology:")
                logs = t_details.get("agent_logs", [])
                if not logs:
                    st.info("No agent logs recorded for this ticket.")
                else:
                    for log in logs:
                        with st.expander(f"🤖 {log['agent_name']} — `{log['step_name']}` ({log['execution_time_ms']:.1f} ms)"):
                            st.write(f"**Status:** `{log['status']}` | **Recorded:** {log['created_at']}")
                            st.write("**Input Data:**")
                            st.json(log["input_data"])
                            st.write("**Output Data:**")
                            st.json(log["output_data"])

                st.markdown("#### Conversation Transcript:")
                for m in t_details.get("messages", []):
                    badge = "👤 Customer" if m["sender"] == "customer" else f"🤖 {m.get('agent_name', 'Agent')}"
                    st.markdown(f"**{badge}** ({m['created_at']}):")
                    st.info(m["content"])

    with tab_metrics:
        st.markdown("### Operational Metrics")
        cats = analytics.get("by_category", {})
        prios = analytics.get("by_priority", {})

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("#### Tickets by Complaint Category")
            if cats:
                st.bar_chart(pd.DataFrame(list(cats.items()), columns=["Category", "Count"]).set_index("Category"))
            else:
                st.info("No category data available.")

        with m_col2:
            st.markdown("#### Tickets by Priority")
            if prios:
                st.bar_chart(pd.DataFrame(list(prios.items()), columns=["Priority", "Count"]).set_index("Priority"))
            else:
                st.info("No priority data available.")

