import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from database.connection import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="customer", nullable=False)  # 'customer' or 'admin'
    phone = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="customer")
    tickets = relationship("Ticket", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String(50), primary_key=True, index=True)  # e.g. "ORD-1001"
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    product_name = Column(String(255), nullable=False)
    item_category = Column(String(100), nullable=False)  # Electronics, Apparel, Books, etc.
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    status = Column(String(50), nullable=False)  # delivered, in_transit, processing, cancelled, returned
    order_date = Column(DateTime, nullable=False)
    delivery_date = Column(DateTime, nullable=True)
    tracking_number = Column(String(100), nullable=True)
    carrier = Column(String(100), default="ExpressShip")
    delivery_address = Column(Text, nullable=True)
    return_window_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    tickets = relationship("Ticket", back_populates="order")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String(50), primary_key=True, index=True)  # e.g. "TCK-5001"
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    order_id = Column(String(50), ForeignKey("orders.id"), nullable=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)  # refund, replacement, cancellation, shipping_delay, damaged_item, general_inquiry
    priority = Column(String(50), default="medium")  # low, medium, high, urgent
    status = Column(String(50), default="open")  # open, investigating, pending_approval, resolved, escalated, closed
    complaint_text = Column(Text, nullable=False)
    resolution_proposal = Column(JSON, nullable=True)  # Structured resolution details
    resolution_summary = Column(Text, nullable=True)
    human_needed = Column(Boolean, default=False)
    escalation_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="tickets")
    order = relationship("Order", back_populates="tickets")
    messages = relationship("ConversationMessage", back_populates="ticket", cascade="all, delete-orphan")
    agent_logs = relationship("AgentLog", back_populates="ticket", cascade="all, delete-orphan")
    approvals = relationship("Approval", back_populates="ticket", cascade="all, delete-orphan")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), ForeignKey("tickets.id"), nullable=False)
    sender = Column(String(50), nullable=False)  # 'customer', 'agent', 'admin', 'system'
    agent_name = Column(String(100), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    ticket = relationship("Ticket", back_populates="messages")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), ForeignKey("tickets.id"), nullable=False)
    agent_name = Column(String(100), nullable=False)
    step_name = Column(String(100), nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    status = Column(String(50), default="success")  # success, error, retry
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    ticket = relationship("Ticket", back_populates="agent_logs")


class Approval(Base):
    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), ForeignKey("tickets.id"), nullable=False)
    action_type = Column(String(100), nullable=False)  # refund, replacement, cancellation, exception_override
    proposed_amount = Column(Float, default=0.0)
    status = Column(String(50), default="pending")  # pending, approved, rejected
    admin_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    admin_notes = Column(Text, nullable=True)
    resolution_applied = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    ticket = relationship("Ticket", back_populates="approvals")

