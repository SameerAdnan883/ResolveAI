import datetime
from sqlalchemy.orm import Session
from database.connection import SessionLocal, Base, engine
from database.models import Customer, Order, Ticket, ConversationMessage, AgentLog, Approval
from services.auth_service import hash_password

def seed_database():
    """Initializes tables and populates realistic synthetic data."""
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        # Check if already seeded
        if db.query(Customer).count() > 0:
            print("Database already seeded.")
            return

        print("Seeding database with customers, orders, and tickets...")

        # 1. Seed Customers (including customer and admin accounts)
        customers_data = [
            Customer(
                id=1,
                email="admin@resolveai.com",
                hashed_password=hash_password("admin123"),
                full_name="Sarah Connor (Lead Admin)",
                role="admin",
                phone="+1-555-0199"
            ),
            Customer(
                id=2,
                email="alice@example.com",
                hashed_password=hash_password("alice123"),
                full_name="Alice Johnson",
                role="customer",
                phone="+1-555-0101"
            ),
            Customer(
                id=3,
                email="bob@example.com",
                hashed_password=hash_password("bob123"),
                full_name="Bob Smith",
                role="customer",
                phone="+1-555-0102"
            ),
            Customer(
                id=4,
                email="charlie@example.com",
                hashed_password=hash_password("charlie123"),
                full_name="Charlie Brown",
                role="customer",
                phone="+1-555-0103"
            ),
            Customer(
                id=5,
                email="diana@example.com",
                hashed_password=hash_password("diana123"),
                full_name="Diana Prince",
                role="customer",
                phone="+1-555-0104"
            )
        ]
        db.add_all(customers_data)
        db.commit()

        # 2. Seed Synthetic Orders with realistic dates and values
        now = datetime.datetime.utcnow()
        orders_data = [
            # Alice's orders (Recent delivery, eligible for refund/replacement)
            Order(
                id="ORD-1001",
                customer_id=2,
                product_name="UltraSound Wireless Noise-Cancelling Headphones",
                item_category="Electronics",
                amount=189.99,
                status="delivered",
                order_date=now - datetime.timedelta(days=10),
                delivery_date=now - datetime.timedelta(days=6),
                tracking_number="EXP-US-8921102",
                carrier="ExpressShip",
                delivery_address="742 Evergreen Terrace, Springfield, IL",
                return_window_days=30
            ),
            Order(
                id="ORD-1002",
                customer_id=2,
                product_name="Organic Cotton Graphic T-Shirt",
                item_category="Apparel",
                amount=29.50,
                status="delivered",
                order_date=now - datetime.timedelta(days=45),
                delivery_date=now - datetime.timedelta(days=40),
                tracking_number="EXP-US-8921103",
                carrier="ExpressShip",
                delivery_address="742 Evergreen Terrace, Springfield, IL",
                return_window_days=30
            ),
            # Bob's orders (One in transit, one processing)
            Order(
                id="ORD-2001",
                customer_id=3,
                product_name="ProBook Mechanical Gaming Keyboard",
                item_category="Electronics",
                amount=129.00,
                status="in_transit",
                order_date=now - datetime.timedelta(days=8),
                delivery_date=None,
                tracking_number="EXP-US-9912034",
                carrier="ExpressShip",
                delivery_address="1042 Elm Street, Dallas, TX",
                return_window_days=30
            ),
            Order(
                id="ORD-2002",
                customer_id=3,
                product_name="Ergonomic Foam Lumbar Cushion",
                item_category="Office Supplies",
                amount=45.00,
                status="processing",
                order_date=now - datetime.timedelta(hours=5),
                delivery_date=None,
                tracking_number=None,
                carrier="StandardPost",
                delivery_address="1042 Elm Street, Dallas, TX",
                return_window_days=30
            ),
            # Charlie's orders (Damaged delivery)
            Order(
                id="ORD-3001",
                customer_id=4,
                product_name="4K OLED 27-inch Gaming Monitor",
                item_category="Electronics",
                amount=399.99,
                status="delivered",
                order_date=now - datetime.timedelta(days=4),
                delivery_date=now - datetime.timedelta(days=1),
                tracking_number="EXP-US-7788190",
                carrier="ExpressShip",
                delivery_address="55 Broadway, New York, NY",
                return_window_days=30
            ),
            # Diana's orders (Recent delivered accessory)
            Order(
                id="ORD-4001",
                customer_id=5,
                product_name="Ceramic Coffee Travel Mug (16oz)",
                item_category="Kitchenware",
                amount=24.99,
                status="delivered",
                order_date=now - datetime.timedelta(days=3),
                delivery_date=now - datetime.timedelta(days=1),
                tracking_number="EXP-US-3344551",
                carrier="StandardPost",
                delivery_address="88 King Street, Seattle, WA",
                return_window_days=30
            )
        ]
        db.add_all(orders_data)
        db.commit()

        # 3. Seed Sample Support Tickets
        tickets_data = [
            Ticket(
                id="TCK-5001",
                customer_id=2,
                order_id="ORD-1001",
                title="Headphones microphone stopped working after 3 days",
                category="replacement",
                priority="high",
                status="pending_approval",
                complaint_text="I bought the UltraSound headphones (ORD-1001) last week. The audio is fine but the microphone is completely dead. I need a replacement.",
                resolution_proposal={
                    "action": "replacement",
                    "reason": "Customer reported hardware defect within 14-day replacement window.",
                    "estimated_dispatch": "1-2 business days",
                    "requires_approval": True,
                    "approval_reason": "Order amount $189.99 exceeds supervisor threshold $150."
                },
                resolution_summary="Proposed expedited replacement under POL-REP-2024-V2. Awaiting supervisor approval.",
                human_needed=True,
                escalation_reason="High value item replacement ($189.99) requires supervisor sign-off."
            ),
            Ticket(
                id="TCK-5002",
                customer_id=3,
                order_id="ORD-2001",
                title="Keyboard delivery delay inquiry",
                category="shipping_delay",
                priority="medium",
                status="investigating",
                complaint_text="My mechanical keyboard ORD-2001 was supposed to arrive 4 days ago. Tracking has not updated since 3 days.",
                resolution_proposal={
                    "action": "carrier_investigation",
                    "reason": "Delivery delayed past SLA; carrier tracking pinged.",
                    "requires_approval": False
                },
                resolution_summary="Carrier trace initiated with ExpressShip.",
                human_needed=False
            ),
            Ticket(
                id="TCK-5003",
                customer_id=2,
                order_id="ORD-1002",
                title="Return request for graphic t-shirt",
                category="refund",
                priority="low",
                status="closed",
                complaint_text="I want to return the cotton t-shirt ORD-1002, bought 45 days ago.",
                resolution_proposal={
                    "action": "rejection",
                    "reason": "Exceeded 30-day return policy window.",
                    "requires_approval": False
                },
                resolution_summary="Ticket closed: Order is past the 30-day return policy window.",
                human_needed=False
            )
        ]
        db.add_all(tickets_data)
        db.commit()

        # 4. Seed an approval record for TCK-5001
        approval_record = Approval(
            id=1,
            ticket_id="TCK-5001",
            action_type="replacement",
            proposed_amount=189.99,
            status="pending",
            resolution_applied=False,
            created_at=now
        )
        db.add(approval_record)

        # 5. Seed Conversation Messages
        messages_data = [
            ConversationMessage(
                ticket_id="TCK-5001",
                sender="customer",
                agent_name=None,
                content="I bought the UltraSound headphones (ORD-1001) last week. The audio is fine but the microphone is completely dead. I need a replacement.",
                created_at=now - datetime.timedelta(hours=2)
            ),
            ConversationMessage(
                ticket_id="TCK-5001",
                sender="agent",
                agent_name="TriageAgent",
                content="Thank you for contacting ResolveAI. I have classified your issue under 'Replacement' with high priority. Our Order Investigation agent is reviewing your order details.",
                created_at=now - datetime.timedelta(hours=2, minutes=-1)
            ),
            ConversationMessage(
                ticket_id="TCK-5001",
                sender="agent",
                agent_name="ResolutionAgent",
                content="We verified your order ORD-1001 delivered 6 days ago. Under our 14-day replacement policy, you qualify for an expedited replacement. Because this order is over $150, our supervisor is conducting a quick final approval.",
                created_at=now - datetime.timedelta(hours=2, minutes=-2)
            )
        ]
        db.add_all(messages_data)

        # 6. Seed Agent Execution Logs
        logs_data = [
            AgentLog(
                ticket_id="TCK-5001",
                agent_name="TriageAgent",
                step_name="classify_intent",
                input_data={"complaint": "Microphone dead on ORD-1001"},
                output_data={"category": "replacement", "priority": "high", "order_id": "ORD-1001"},
                execution_time_ms=145.2,
                status="success"
            ),
            AgentLog(
                ticket_id="TCK-5001",
                agent_name="OrderAgent",
                step_name="fetch_order_details",
                input_data={"order_id": "ORD-1001"},
                output_data={"found": True, "amount": 189.99, "status": "delivered", "delivered_days_ago": 6},
                execution_time_ms=52.8,
                status="success"
            ),
            AgentLog(
                ticket_id="TCK-5001",
                agent_name="PolicyAgent",
                step_name="search_policy",
                input_data={"query": "defective replacement window"},
                output_data={"policy_id": "POL-REP-2024-V2", "clause": "1.2: 14 calendar days"},
                execution_time_ms=89.1,
                status="success"
            )
        ]
        db.add_all(logs_data)
        db.commit()

        print("Database seeded successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()

