import streamlit as st
from frontend.api_client import ResolveAIClient

def render_customer_portal(client: ResolveAIClient):
    st.markdown("## 🛒 Customer Resolution Portal")
    st.caption("Submit complaints, track orders, and experience autonomous multi-agent issue resolution.")

    # Check authentication
    if "customer_token" not in st.session_state:
        st.session_state.customer_token = None
    if "customer_user" not in st.session_state:
        st.session_state.customer_user = None

    if not st.session_state.customer_token:
        tab_login, tab_register = st.tabs(["Customer Login", "Create Account"])

        with tab_login:
            st.markdown("### Sign In to Your Customer Account")
            st.info("Demo Account: `alice@example.com` / Password: `alice123`")
            email = st.text_input("Email", value="alice@example.com", key="cust_login_email")
            password = st.text_input("Password", value="alice123", type="password", key="cust_login_pass")
            if st.button("Sign In", type="primary", key="cust_btn_login"):
                res = client.login(email, password)
                if "access_token" in res:
                    st.session_state.customer_token = res["access_token"]
                    st.session_state.customer_user = res["user"]
                    st.success(f"Welcome back, {res['user']['full_name']}!")
                    st.rerun()
                else:
                    st.error(res.get("error", "Login failed."))

        with tab_register:
            st.markdown("### Create New Customer Account")
            reg_name = st.text_input("Full Name", key="cust_reg_name")
            reg_email = st.text_input("Email", key="cust_reg_email")
            reg_pass = st.text_input("Password", type="password", key="cust_reg_pass")
            if st.button("Register Account", key="cust_btn_reg"):
                res = client.register(reg_email, reg_pass, reg_name)
                if "access_token" in res:
                    st.session_state.customer_token = res["access_token"]
                    st.session_state.customer_user = res["user"]
                    st.success("Account created successfully!")
                    st.rerun()
                else:
                    st.error(res.get("error", "Registration failed."))
        return

    # Authenticated Customer Dashboard
    user = st.session_state.customer_user
    token = st.session_state.customer_token

    col_user, col_logout = st.columns([4, 1])
    with col_user:
        st.markdown(f"**Logged in as:** {user['full_name']} (`{user['email']}`)")
    with col_logout:
        if st.button("Sign Out", key="cust_logout"):
            st.session_state.customer_token = None
            st.session_state.customer_user = None
            st.rerun()

    st.divider()

    portal_tab1, portal_tab2 = st.tabs(["Submit New Issue", "My Support Tickets"])

    with portal_tab1:
        st.markdown("### 💬 Submit an Order Issue or Support Request")
        st.markdown(
            "Our autonomous multi-agent system (`Triage`, `Order Investigation`, `Policy Research`, `Resolution`, and `Quality`) will analyze your request in real time."
        )

        col_ord, col_blank = st.columns([2, 2])
        with col_ord:
            order_id_input = st.text_input(
                "Order ID (Optional if mentioned in complaint text):",
                placeholder="e.g. ORD-1001",
                help="Reference your order from the synthetic order catalog"
            )

        complaint_input = st.text_area(
            "Describe your issue in detail:",
            placeholder="e.g. I bought the UltraSound headphones (ORD-1001) last week, but the microphone is completely dead. I would like a replacement or refund.",
            height=120
        )

        # Quick sample prompt buttons
        st.caption("Quick Test Scenarios:")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("Defective Item (ORD-1001)"):
                st.session_state.quick_complaint = "The headphones microphone in ORD-1001 stopped working completely after 3 days. I need an expedited replacement."
                st.rerun()
        with col_q2:
            if st.button("Expired Return (ORD-1002)"):
                st.session_state.quick_complaint = "I want to return the graphic t-shirt ORD-1002 bought 45 days ago for a full refund."
                st.rerun()
        with col_q3:
            if st.button("Cancel In-Transit (ORD-2001)"):
                st.session_state.quick_complaint = "Please cancel my keyboard order ORD-2001 immediately. I changed my mind."
                st.rerun()

        if "quick_complaint" in st.session_state:
            complaint_input = st.session_state.pop("quick_complaint")

        if st.button("Submit to Multi-Agent Resolution Team", type="primary", key="cust_submit_btn"):
            if not complaint_input.strip():
                st.warning("Please describe your issue first.")
            else:
                with st.spinner("🤖 Autonomous AI Agents are collaborating on your ticket..."):
                    res = client.submit_complaint(token, complaint_input, order_id_input)

                if "error" in res:
                    st.error(f"Error submitting ticket: {res['error']}")
                else:
                    st.success(f"Ticket Created: **{res.get('ticket_id')}** (Status: **{res.get('status')}**)")

                    # Multi-Agent Execution Breakdown Visualizer
                    st.markdown("#### 🔍 Multi-Agent Execution Walkthrough")
                    
                    with st.expander("Agent 1: Triage & Intent Classification", expanded=True):
                        st.write(f"- **Category:** `{res.get('category')}`")
                        st.write(f"- **Priority:** `{res.get('priority')}`")
                        st.write(f"- **Extracted Order ID:** `{res.get('extracted_order_id')}`")

                    with st.expander("Agent 2 & 3: Order Investigation & Policy Retrieval", expanded=True):
                        st.write(f"- **Order Record Found in DB:** `{res.get('order_found')}`")
                        st.info("Policy knowledge retrieved from ChromaDB vector store.")

                    with st.expander("Agent 4 & 5: Proposed Resolution & Audit", expanded=True):
                        proposal = res.get("proposed_resolution", {})
                        st.write(f"- **Proposed Action:** `{proposal.get('action')}`")
                        st.write(f"- **Eligible:** `{proposal.get('eligible')}`")
                        st.write(f"- **Human Approval Required:** `{res.get('human_approval_required')}`")
                        st.write(f"- **Policy Reference:** `{proposal.get('policy_cited')}`")
                        st.success(f"**Customer Message:** {proposal.get('customer_message')}")

    with portal_tab2:
        st.markdown("### 📋 Your Support History")
        tickets = client.get_my_tickets(token)
        if not tickets:
            st.info("You haven't submitted any support tickets yet.")
        else:
            for t in tickets:
                with st.expander(f"🎫 {t['id']} — {t['title']} ({t['status'].upper()})", expanded=False):
                    st.write(f"**Complaint:** {t['complaint_text']}")
                    st.write(f"**Category:** `{t['category']}` | **Priority:** `{t['priority']}`")
                    st.write(f"**Created:** {t['created_at']}")
                    if t.get("resolution_summary"):
                        st.info(f"**Resolution:** {t['resolution_summary']}")
                    if t.get("human_needed"):
                        st.warning("⚠️ This ticket is currently escalated and undergoing supervisor review.")

