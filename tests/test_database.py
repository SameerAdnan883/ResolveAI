import pytest
from database.models import Customer, Order, Ticket, Approval, AgentLog
from database.seed_data import seed_database
from database.connection import SessionLocal

def test_database_seeding_and_models():
    seed_database()
    db = SessionLocal()
    try:
        customers = db.query(Customer).all()
        assert len(customers) >= 5
        admin = db.query(Customer).filter(Customer.email == "admin@resolveai.com").first()
        assert admin is not None
        assert admin.role == "admin"

        orders = db.query(Order).all()
        assert len(orders) >= 6
        order_1001 = db.query(Order).filter(Order.id == "ORD-1001").first()
        assert order_1001 is not None
        assert order_1001.amount == 189.99

        tickets = db.query(Ticket).all()
        assert len(tickets) >= 3

        approvals = db.query(Approval).all()
        assert len(approvals) >= 1
    finally:
        db.close()

