"""Initial VERIFLOW persistence layer for companies and users."""

from __future__ import annotations

import secrets
from datetime import date, datetime, timezone
from decimal import Decimal

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    joinedload,
    mapped_column,
    relationship,
    sessionmaker,
)


DATABASE_URL = "sqlite:///veriflow.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(connection, _record) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
_password_hasher = PasswordHasher()  # Argon2id with argon2-cffi's secure defaults.
VALID_ROLES = frozenset({"admin", "finance", "viewer"})
DUMMY_PASSWORD_HASH = _password_hasher.hash(secrets.token_urlsafe(48))
INVOICE_STATUSES = frozenset({"pending", "approved", "rejected"})
APPROVAL_DECISIONS = frozenset({"approved", "rejected"})


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    secondary_approval_threshold: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        default=Decimal("10000.00"),
        server_default=text("10000.00"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    users: Mapped[list[User]] = relationship(back_populates="company")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'finance', 'viewer')",
            name="ck_users_role",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="viewer",
        server_default=text("'viewer'"),
    )
    is_finance_manager: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    company: Mapped[Company] = relationship(back_populates="users")


class Supplier(Base):
    __tablename__ = "suppliers"
    __table_args__ = (
        UniqueConstraint("company_id", "name", name="uq_suppliers_company_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    tax_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    invoices: Mapped[list[Invoice]] = relationship(back_populates="supplier")


class Invoice(Base):
    __tablename__ = "invoices"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_invoices_status",
        ),
        CheckConstraint("amount > 0", name="ck_invoices_positive_amount"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), nullable=False, index=True
    )
    supplier_id: Mapped[int] = mapped_column(
        ForeignKey("suppliers.id"), nullable=False, index=True
    )
    invoice_number: Mapped[str] = mapped_column(String(100), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    issue_date: Mapped[date] = mapped_column(Date, nullable=False)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    description: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    approval_threshold_used: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        default=Decimal("10000.00"),
        server_default=text("10000.00"),
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default=text("'pending'")
    )
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    supplier: Mapped[Supplier] = relationship(back_populates="invoices")
    documents: Mapped[list[InvoiceDocument]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    approvals: Mapped[list[ApprovalEvent]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )


class InvoiceDocument(Base):
    __tablename__ = "invoice_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), nullable=False, index=True
    )
    invoice_id: Mapped[int] = mapped_column(
        ForeignKey("invoices.id"), nullable=False, index=True
    )
    storage_key: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    uploaded_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    invoice: Mapped[Invoice] = relationship(back_populates="documents")


class ApprovalEvent(Base):
    __tablename__ = "approval_events"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('approved', 'rejected')",
            name="ck_approval_events_decision",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), nullable=False, index=True
    )
    invoice_id: Mapped[int] = mapped_column(
        ForeignKey("invoices.id"), nullable=False, index=True
    )
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    approval_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    note: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    invoice: Mapped[Invoice] = relationship(back_populates="approvals")
    actor: Mapped[User] = relationship()


def init_db() -> None:
    """Create tables and add new columns safely to existing SQLite databases."""
    Base.metadata.create_all(bind=engine)
    migrations = {
        "users": {
            "role": "VARCHAR(20) NOT NULL DEFAULT 'viewer'",
            "is_finance_manager": "BOOLEAN NOT NULL DEFAULT 0",
        },
        "companies": {
            "secondary_approval_threshold": (
                "NUMERIC(14, 2) NOT NULL DEFAULT 10000.00"
            ),
        },
        "invoices": {
            "approval_threshold_used": "NUMERIC(14, 2) NOT NULL DEFAULT 10000.00",
        },
        "approval_events": {
            "approval_level": "INTEGER NOT NULL DEFAULT 1",
        },
    }
    with engine.begin() as connection:
        for table_name, columns in migrations.items():
            existing_columns = {
                column["name"] for column in inspect(connection).get_columns(table_name)
            }
            for column_name, definition in columns.items():
                if column_name not in existing_columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN {column_name} {definition}"
                        )
                    )


def hash_password(password: str) -> str:
    """Hash a password with Argon2id; plaintext passwords are never stored."""
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Password must be at least 12 characters long.")
    if len(password) > 1024:
        raise ValueError("Password is too long.")
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its stored Argon2id hash."""
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerificationError, TypeError):
        return False


def create_company(name: str) -> Company:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Company name is required.")
    if len(clean_name) > 200:
        raise ValueError("Company name must be 200 characters or fewer.")

    with SessionLocal() as session:
        company = Company(name=clean_name)
        session.add(company)
        session.commit()
        session.refresh(company)
        return company


def register_company_account(company_name: str, email: str, password: str) -> User:
    """Atomically register a new company and its first admin account."""
    clean_name = company_name.strip()
    clean_email = email.strip().lower()
    if not clean_name or len(clean_name) > 200:
        raise ValueError("Company name must contain 1 to 200 characters.")
    if not clean_email or len(clean_email) > 320 or "@" not in clean_email:
        raise ValueError("A valid email address is required.")

    password_hash = hash_password(password)
    with SessionLocal() as session:
        company = Company(name=clean_name)
        session.add(company)
        session.flush()
        user = User(
            company_id=company.id,
            email=clean_email,
            password_hash=password_hash,
            role="admin",
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("An account with this email address already exists.") from None
        session.refresh(user)
        statement = select(User).options(joinedload(User.company)).where(User.id == user.id)
        return session.scalar(statement)


def create_user(
    company_id: int,
    email: str,
    password: str,
    role: str = "viewer",
    is_finance_manager: bool = False,
) -> User:
    clean_email = email.strip().lower()
    if not clean_email or len(clean_email) > 320 or "@" not in clean_email:
        raise ValueError("A valid email address is required.")
    if role not in VALID_ROLES:
        raise ValueError("Role must be admin, finance, or viewer.")
    if is_finance_manager and role != "finance":
        raise ValueError("A finance manager must have the finance role.")

    user = User(
        company_id=company_id,
        email=clean_email,
        password_hash=hash_password(password),
        role=role,
        is_finance_manager=is_finance_manager,
    )
    with SessionLocal() as session:
        if session.get(Company, company_id) is None:
            raise ValueError("Company was not found.")
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("A user with this email address already exists.") from None
        session.refresh(user)
        return user


def get_user_by_email(email: str) -> User | None:
    clean_email = email.strip().lower()
    with SessionLocal() as session:
        statement = (
            select(User)
            .options(joinedload(User.company))
            .where(User.email == clean_email)
        )
        return session.scalar(statement)


def get_user_by_id(user_id: int) -> User | None:
    """Load the signed-in user and their company from trusted database data."""
    with SessionLocal() as session:
        statement = (
            select(User)
            .options(joinedload(User.company))
            .where(User.id == user_id)
        )
        return session.scalar(statement)


def get_user_by_id_for_company(user_id: int, company_id: int) -> User | None:
    """Load a user only when they belong to the caller's trusted company scope.

    Company-owned API resources should use this lookup rather than relying on
    a user ID alone when resolving a user supplied by a request.
    """
    with SessionLocal() as session:
        statement = (
            select(User)
            .options(joinedload(User.company))
            .where(User.id == user_id, User.company_id == company_id)
        )
        return session.scalar(statement)
