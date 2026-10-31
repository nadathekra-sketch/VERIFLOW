"""Company-scoped finance operations used by the VERIFLOW Streamlit app."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, selectinload

from database import (
    APPROVAL_DECISIONS,
    INVOICE_STATUSES,
    SessionLocal,
    ApprovalEvent,
    Company,
    Invoice,
    InvoiceDocument,
    Supplier,
    User,
    hash_password,
)


UPLOAD_ROOT = Path("veriflow_uploads")
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
FINANCE_ROLES = frozenset({"admin", "finance"})
DOCUMENT_TYPES = {
    ".pdf": ("application/pdf", b"%PDF-"),
    ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
    ".jpg": ("image/jpeg", b"\xff\xd8\xff"),
    ".jpeg": ("image/jpeg", b"\xff\xd8\xff"),
}
STORED_KEY_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|png|jpg)$")


@dataclass(frozen=True)
class DashboardStats:
    total_invoices: int
    total_suppliers: int
    missing_documents: int
    pending_approvals: int


@dataclass(frozen=True)
class PaymentReviewItem:
    invoice: Invoice
    missing_documents: bool
    duplicate_invoice_ids: tuple[int, ...]
    current_approval_level: int
    required_approval_levels: int


@dataclass(frozen=True)
class MonthlySpend:
    month: str
    amount: Decimal


@dataclass(frozen=True)
class SupplierSpend:
    supplier_name: str
    amount: Decimal


@dataclass(frozen=True)
class DashboardAnalytics:
    currency: str
    monthly_spend: tuple[MonthlySpend, ...]
    supplier_spend: tuple[SupplierSpend, ...]


@dataclass(frozen=True)
class NotificationItem:
    key: str
    title: str
    message: str
    created_at: datetime
    invoice_id: int


@dataclass(frozen=True)
class TeamMember:
    id: int
    email: str
    role: str
    is_finance_manager: bool
    created_at: datetime


def _require_company_admin(session, company_id: int, user_id: int) -> User:
    user = session.scalar(
        select(User).where(User.id == user_id, User.company_id == company_id)
    )
    if user is None:
        raise ValueError("User was not found in this company.")
    if user.role != "admin":
        raise PermissionError("Only a company administrator can change approval settings.")
    return user


def _require_finance_user(session, company_id: int, user_id: int) -> User:
    user = session.scalar(
        select(User).where(User.id == user_id, User.company_id == company_id)
    )
    if user is None:
        raise ValueError("User was not found in this company.")
    if user.role not in FINANCE_ROLES:
        raise PermissionError("This role cannot modify finance records.")
    return user


def _is_finance_manager(user: User) -> bool:
    return user.role == "admin" or user.is_finance_manager


def _team_member_view(user: User) -> TeamMember:
    return TeamMember(
        id=user.id,
        email=user.email,
        role=user.role,
        is_finance_manager=user.is_finance_manager,
        created_at=user.created_at,
    )


def create_company_member(
    company_id: int,
    actor_user_id: int,
    email: str,
    password: str,
    role: str,
    is_finance_manager: bool = False,
) -> TeamMember:
    clean_email = email.strip().lower()
    if not clean_email or len(clean_email) > 320 or "@" not in clean_email:
        raise ValueError("A valid email address is required.")
    if role not in {"finance", "viewer"}:
        raise ValueError("New company accounts must be finance or viewer users.")
    if is_finance_manager and role != "finance":
        raise ValueError("A finance manager must have the finance role.")

    with SessionLocal() as session:
        _require_company_admin(session, company_id, actor_user_id)
        user = User(
            company_id=company_id,
            email=clean_email,
            password_hash=hash_password(password),
            role=role,
            is_finance_manager=is_finance_manager,
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("A user with this email address already exists.") from None
        session.refresh(user)
        return _team_member_view(user)


def list_company_members(company_id: int, actor_user_id: int) -> list[TeamMember]:
    with SessionLocal() as session:
        _require_company_admin(session, company_id, actor_user_id)
        users = session.scalars(
            select(User)
            .where(User.company_id == company_id)
            .order_by(User.created_at, User.id)
        ).all()
        return [_team_member_view(user) for user in users]


def get_company_approval_threshold(company_id: int) -> Decimal:
    with SessionLocal() as session:
        company = session.get(Company, company_id)
        if company is None:
            raise ValueError("Company was not found.")
        return company.secondary_approval_threshold


def set_company_approval_threshold(
    company_id: int,
    actor_user_id: int,
    threshold: Decimal | int | float | str,
) -> Decimal:
    try:
        normalized = Decimal(str(threshold)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):
        raise ValueError("Approval threshold is invalid.") from None
    if (
        not normalized.is_finite()
        or normalized <= 0
        or normalized > Decimal("999999999999.99")
    ):
        raise ValueError("Approval threshold must be greater than zero.")

    with SessionLocal() as session:
        _require_company_admin(session, company_id, actor_user_id)
        company = session.get(Company, company_id)
        if company is None:
            raise ValueError("Company was not found.")
        company.secondary_approval_threshold = normalized
        session.commit()
        return normalized


def _clean_optional(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        raise ValueError(f"Value must be {limit} characters or fewer.")
    return cleaned


def _clean_supplier_fields(
    name: str,
    email: str | None,
    tax_id: str | None,
) -> tuple[str, str | None, str | None]:
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 200:
        raise ValueError("Supplier name must contain 1 to 200 characters.")
    clean_email = _clean_optional(email, 320)
    if clean_email is not None:
        clean_email = clean_email.lower()
        if "@" not in clean_email:
            raise ValueError("Supplier email address is invalid.")
    clean_tax_id = _clean_optional(tax_id, 100)
    return clean_name, clean_email, clean_tax_id


def create_supplier(
    company_id: int,
    actor_user_id: int,
    name: str,
    email: str | None = None,
    tax_id: str | None = None,
) -> Supplier:
    clean_name, clean_email, clean_tax_id = _clean_supplier_fields(name, email, tax_id)
    with SessionLocal() as session:
        _require_finance_user(session, company_id, actor_user_id)
        duplicate = session.scalar(
            select(Supplier.id).where(
                Supplier.company_id == company_id,
                func.lower(Supplier.name) == clean_name.lower(),
            )
        )
        if duplicate is not None:
            raise ValueError("A supplier with this name already exists.")
        supplier = Supplier(
            company_id=company_id,
            name=clean_name,
            email=clean_email,
            tax_id=clean_tax_id,
        )
        session.add(supplier)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("A supplier with this name already exists.") from None
        session.refresh(supplier)
        return supplier


def update_supplier(
    company_id: int,
    actor_user_id: int,
    supplier_id: int,
    name: str,
    email: str | None,
    tax_id: str | None,
    active: bool,
) -> Supplier:
    clean_name, clean_email, clean_tax_id = _clean_supplier_fields(name, email, tax_id)
    with SessionLocal() as session:
        _require_finance_user(session, company_id, actor_user_id)
        company = session.get(Company, company_id)
        if company is None:
            raise ValueError("Company was not found.")
        supplier = session.scalar(
            select(Supplier).where(
                Supplier.id == supplier_id,
                Supplier.company_id == company_id,
            )
        )
        if supplier is None:
            raise ValueError("Supplier was not found in this company.")
        duplicate = session.scalar(
            select(Supplier.id).where(
                Supplier.company_id == company_id,
                Supplier.id != supplier_id,
                func.lower(Supplier.name) == clean_name.lower(),
            )
        )
        if duplicate is not None:
            raise ValueError("A supplier with this name already exists.")
        supplier.name = clean_name
        supplier.email = clean_email
        supplier.tax_id = clean_tax_id
        supplier.active = bool(active)
        supplier.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(supplier)
        return supplier


def list_suppliers(company_id: int, active_only: bool = False) -> list[Supplier]:
    with SessionLocal() as session:
        statement = select(Supplier).where(Supplier.company_id == company_id)
        if active_only:
            statement = statement.where(Supplier.active.is_(True))
        return list(session.scalars(statement.order_by(Supplier.name)).all())


def list_supplier_invoices(company_id: int, supplier_id: int) -> list[Invoice]:
    with SessionLocal() as session:
        supplier_exists = session.scalar(
            select(Supplier.id).where(
                Supplier.id == supplier_id,
                Supplier.company_id == company_id,
            )
        )
        if supplier_exists is None:
            raise ValueError("Supplier was not found in this company.")
        statement = (
            select(Invoice)
            .options(joinedload(Invoice.supplier), selectinload(Invoice.documents))
            .join(
                Supplier,
                and_(
                    Supplier.id == Invoice.supplier_id,
                    Supplier.company_id == Invoice.company_id,
                ),
            )
            .where(
                Invoice.company_id == company_id,
                Invoice.supplier_id == supplier_id,
                Supplier.company_id == company_id,
            )
            .order_by(Invoice.issue_date.desc(), Invoice.id.desc())
        )
        return list(session.scalars(statement).all())


def _normalize_amount(amount: Decimal | int | float | str) -> Decimal:
    try:
        normalized = Decimal(str(amount)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):
        raise ValueError("Invoice amount is invalid.") from None
    if not normalized.is_finite() or normalized <= 0 or normalized > Decimal("999999999999.99"):
        raise ValueError("Invoice amount must be greater than zero and within the supported limit.")
    return normalized


def create_invoice(
    company_id: int,
    actor_user_id: int,
    supplier_id: int,
    invoice_number: str,
    amount: Decimal | int | float | str,
    currency: str,
    issue_date: date,
    due_date: date | None = None,
    description: str = "",
) -> Invoice:
    clean_number = invoice_number.strip()
    if not clean_number or len(clean_number) > 100:
        raise ValueError("Invoice number must contain 1 to 100 characters.")
    clean_currency = currency.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", clean_currency):
        raise ValueError("Currency must be a three-letter code.")
    if not isinstance(issue_date, date):
        raise ValueError("Invoice issue date is required.")
    if due_date is not None and due_date < issue_date:
        raise ValueError("Due date cannot be earlier than the issue date.")
    clean_description = description.strip()
    if len(clean_description) > 1000:
        raise ValueError("Description must be 1000 characters or fewer.")
    normalized_amount = _normalize_amount(amount)

    with SessionLocal() as session:
        _require_finance_user(session, company_id, actor_user_id)
        supplier = session.scalar(
            select(Supplier).where(
                Supplier.id == supplier_id,
                Supplier.company_id == company_id,
                Supplier.active.is_(True),
            )
        )
        if supplier is None:
            raise ValueError("Active supplier was not found in this company.")
        invoice = Invoice(
            company_id=company_id,
            supplier_id=supplier.id,
            invoice_number=clean_number,
            amount=normalized_amount,
            currency=clean_currency,
            issue_date=issue_date,
            due_date=due_date,
            description=clean_description,
            approval_threshold_used=company.secondary_approval_threshold,
            status="pending",
            created_by_user_id=actor_user_id,
        )
        session.add(invoice)
        session.commit()
        session.refresh(invoice)
        return invoice


def list_invoices(
    company_id: int,
    search: str = "",
    status: str | None = None,
    supplier_id: int | None = None,
) -> list[Invoice]:
    if status not in (None, "all", *INVOICE_STATUSES):
        raise ValueError("Invoice status is invalid.")
    statement = (
        select(Invoice)
        .options(
            joinedload(Invoice.supplier),
            selectinload(Invoice.documents),
            selectinload(Invoice.approvals),
        )
        .join(
            Supplier,
            and_(
                Supplier.id == Invoice.supplier_id,
                Supplier.company_id == Invoice.company_id,
            ),
        )
        .where(
            Invoice.company_id == company_id,
            Supplier.company_id == company_id,
        )
    )
    if status not in (None, "all"):
        statement = statement.where(Invoice.status == status)
    if supplier_id is not None:
        statement = statement.where(Invoice.supplier_id == supplier_id)
    clean_search = search.strip()
    if clean_search:
        pattern = f"%{clean_search}%"
        statement = statement.where(
            or_(
                Invoice.invoice_number.ilike(pattern),
                Supplier.name.ilike(pattern),
                Invoice.description.ilike(pattern),
            )
        )
    statement = statement.order_by(Invoice.issue_date.desc(), Invoice.id.desc())
    with SessionLocal() as session:
        return list(session.scalars(statement).all())


def get_dashboard_stats(company_id: int) -> DashboardStats:
    with SessionLocal() as session:
        total_invoices = session.scalar(
            select(func.count(Invoice.id)).where(Invoice.company_id == company_id)
        )
        total_suppliers = session.scalar(
            select(func.count(Supplier.id)).where(Supplier.company_id == company_id)
        )
        missing_documents = session.scalar(
            select(func.count(Invoice.id)).where(
                Invoice.company_id == company_id,
                ~Invoice.documents.any(),
            )
        )
        pending_approvals = session.scalar(
            select(func.count(Invoice.id)).where(
                Invoice.company_id == company_id,
                Invoice.status == "pending",
            )
        )
    return DashboardStats(
        total_invoices=total_invoices or 0,
        total_suppliers=total_suppliers or 0,
        missing_documents=missing_documents or 0,
        pending_approvals=pending_approvals or 0,
    )


def get_analytics_currencies(company_id: int) -> list[str]:
    with SessionLocal() as session:
        currencies = session.scalars(
            select(Invoice.currency)
            .where(
                Invoice.company_id == company_id,
                Invoice.status == "approved",
            )
            .distinct()
            .order_by(Invoice.currency)
        ).all()
        return list(currencies)


def _shift_month(month_start: date, offset: int) -> date:
    month_index = month_start.year * 12 + month_start.month - 1 + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def get_dashboard_analytics(
    company_id: int,
    currency: str,
    month_count: int = 12,
    as_of: date | None = None,
) -> DashboardAnalytics:
    clean_currency = currency.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", clean_currency):
        raise ValueError("Currency must be a three-letter code.")
    if not 1 <= month_count <= 36:
        raise ValueError("Analytics period must be between 1 and 36 months.")

    end_month = (as_of or date.today()).replace(day=1)
    start_month = _shift_month(end_month, 1 - month_count)
    next_month = _shift_month(end_month, 1)
    with SessionLocal() as session:
        monthly_totals = session.execute(
            select(
                func.strftime("%Y-%m", Invoice.issue_date),
                func.sum(Invoice.amount),
            )
            .where(
                Invoice.company_id == company_id,
                Invoice.status == "approved",
                Invoice.currency == clean_currency,
                Invoice.issue_date >= start_month,
                Invoice.issue_date < next_month,
            )
            .group_by(func.strftime("%Y-%m", Invoice.issue_date))
        ).all()
        supplier_totals = session.execute(
            select(Supplier.name, func.sum(Invoice.amount))
            .join(
                Supplier,
                and_(
                    Supplier.id == Invoice.supplier_id,
                    Supplier.company_id == Invoice.company_id,
                ),
            )
            .where(
                Invoice.company_id == company_id,
                Supplier.company_id == company_id,
                Invoice.status == "approved",
                Invoice.currency == clean_currency,
                Invoice.issue_date >= start_month,
                Invoice.issue_date < next_month,
            )
            .group_by(Supplier.id, Supplier.name)
            .order_by(func.sum(Invoice.amount).desc(), Supplier.name)
        ).all()

    totals_by_month = {
        str(month): Decimal(str(amount or 0)) for month, amount in monthly_totals
    }
    monthly_spend = tuple(
        MonthlySpend(
            month=month.strftime("%b %Y"),
            amount=totals_by_month.get(month.strftime("%Y-%m"), Decimal("0.00")),
        )
        for month in (
            _shift_month(start_month, offset) for offset in range(month_count)
        )
    )
    return DashboardAnalytics(
        currency=clean_currency,
        monthly_spend=monthly_spend,
        supplier_spend=tuple(
            SupplierSpend(name, Decimal(str(amount or 0)))
            for name, amount in supplier_totals
        ),
    )


def _duplicate_invoice_ids(session, invoice: Invoice) -> tuple[int, ...]:
    normalized_number = invoice.invoice_number.strip().lower()
    statement = select(Invoice.id).where(
        Invoice.company_id == invoice.company_id,
        Invoice.supplier_id == invoice.supplier_id,
        Invoice.id != invoice.id,
        or_(
            func.lower(func.trim(Invoice.invoice_number)) == normalized_number,
            and_(
                Invoice.amount == invoice.amount,
                Invoice.issue_date == invoice.issue_date,
            ),
        ),
    )
    return tuple(sorted(session.scalars(statement).all()))


def get_payment_review_items(company_id: int) -> list[PaymentReviewItem]:
    invoices = list_invoices(company_id)
    with SessionLocal() as session:
        items: list[PaymentReviewItem] = []
        for invoice in invoices:
            if invoice.status != "pending":
                continue
            duplicate_ids = _duplicate_invoice_ids(session, invoice)
            approved_levels = sorted(
                approval.approval_level
                for approval in invoice.approvals
                if approval.decision == "approved"
            )
            required_levels = (
                2
                if invoice.amount > invoice.approval_threshold_used
                else 1
            )
            items.append(
                PaymentReviewItem(
                    invoice=invoice,
                    missing_documents=not invoice.documents,
                    duplicate_invoice_ids=duplicate_ids,
                    current_approval_level=len(approved_levels) + 1,
                    required_approval_levels=required_levels,
                )
            )
        return items


def get_recent_notifications(
    company_id: int, limit: int = 15
) -> list[NotificationItem]:
    if not 1 <= limit <= 50:
        raise ValueError("Notification limit must be between 1 and 50.")

    notifications: list[NotificationItem] = []
    for item in get_payment_review_items(company_id):
        invoice = item.invoice
        supplier_name = invoice.supplier.name if invoice.supplier else "Supplier"
        if item.missing_documents:
            notifications.append(
                NotificationItem(
                    key=f"missing-document-{invoice.id}",
                    title="Missing document",
                    message=f"Invoice {invoice.invoice_number} from {supplier_name} "
                    "needs a supporting document.",
                    created_at=invoice.updated_at,
                    invoice_id=invoice.id,
                )
            )
        if item.duplicate_invoice_ids:
            notifications.append(
                NotificationItem(
                    key=f"possible-duplicate-{invoice.id}",
                    title="Potential duplicate",
                    message=f"Invoice {invoice.invoice_number} from {supplier_name} "
                    "matches another invoice.",
                    created_at=invoice.updated_at,
                    invoice_id=invoice.id,
                )
            )

    with SessionLocal() as session:
        events = session.execute(
            select(ApprovalEvent, Invoice, User)
            .join(Invoice, Invoice.id == ApprovalEvent.invoice_id)
            .join(User, User.id == ApprovalEvent.actor_user_id)
            .where(
                ApprovalEvent.company_id == company_id,
                Invoice.company_id == company_id,
            )
            .order_by(ApprovalEvent.created_at.desc(), ApprovalEvent.id.desc())
            .limit(limit)
        ).all()
        for approval, invoice, actor in events:
            if approval.decision == "rejected":
                title = "Invoice rejected"
                detail = f"Rejected by {actor.email}."
                if approval.note:
                    detail = f"{detail} Note: {approval.note}"
            elif (
                approval.approval_level == 1
                and invoice.status == "pending"
                and invoice.amount > invoice.approval_threshold_used
            ):
                title = "First approval recorded"
                detail = (
                    f"Approved by {actor.email}; finance manager review is still required."
                )
            else:
                title = "Invoice approved"
                detail = (
                    f"Approval level {approval.approval_level} completed by {actor.email}."
                )
            notifications.append(
                NotificationItem(
                    key=f"approval-event-{approval.id}",
                    title=title,
                    message=f"Invoice {invoice.invoice_number}. {detail}",
                    created_at=approval.created_at,
                    invoice_id=invoice.id,
                )
            )

    return sorted(
        notifications,
        key=lambda notification: notification.created_at,
        reverse=True,
    )[:limit]


def record_invoice_decision(
    company_id: int,
    actor_user_id: int,
    invoice_id: int,
    decision: str,
    note: str = "",
) -> Invoice:
    if decision not in APPROVAL_DECISIONS:
        raise ValueError("Decision must be approved or rejected.")
    clean_note = note.strip()
    if len(clean_note) > 1000:
        raise ValueError("Decision note must be 1000 characters or fewer.")

    with SessionLocal() as session:
        actor = _require_finance_user(session, company_id, actor_user_id)
        invoice = session.scalar(
            select(Invoice)
            .options(selectinload(Invoice.documents))
            .where(
                Invoice.id == invoice_id,
                Invoice.company_id == company_id,
            )
        )
        if invoice is None:
            raise ValueError("Invoice was not found in this company.")
        if invoice.status != "pending":
            raise ValueError("Only pending invoices can be reviewed.")
        approved_events = sorted(
            (
                approval
                for approval in invoice.approvals
                if approval.decision == "approved"
            ),
            key=lambda approval: approval.approval_level,
        )
        current_level = len(approved_events) + 1
        required_levels = (
            2 if invoice.amount > invoice.approval_threshold_used else 1
        )
        if current_level > required_levels:
            raise ValueError("The invoice has already completed its approval chain.")
        if current_level == 2:
            if not _is_finance_manager(actor):
                raise PermissionError(
                    "A finance manager must complete the secondary approval."
                )
            if any(approval.actor_user_id == actor.id for approval in approved_events):
                raise PermissionError(
                    "The secondary approval must come from a different reviewer."
                )
        if decision == "approved":
            if invoice.created_by_user_id == actor.id:
                raise PermissionError("The invoice creator cannot approve their own invoice.")
            if not invoice.documents:
                raise ValueError("Attach at least one document before approval.")
            if _duplicate_invoice_ids(session, invoice):
                raise ValueError("Resolve the potential duplicate before approval.")

        if decision == "rejected":
            invoice.status = "rejected"
        elif current_level == required_levels:
            invoice.status = "approved"
        invoice.updated_at = datetime.now(timezone.utc)
        session.add(
            ApprovalEvent(
                company_id=company_id,
                invoice_id=invoice.id,
                actor_user_id=actor.id,
                decision=decision,
                approval_level=current_level,
                note=clean_note,
            )
        )
        session.commit()
        session.refresh(invoice)
        return invoice


def _safe_upload_name(original_filename: str) -> tuple[str, str, str]:
    filename = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
    filename = "".join(char for char in filename if char.isprintable()).strip()
    if not filename or len(filename) > 255:
        raise ValueError("Document filename must contain 1 to 255 printable characters.")
    extension = Path(filename).suffix.lower()
    document_type = DOCUMENT_TYPES.get(extension)
    if document_type is None:
        raise ValueError("Only PDF, PNG, and JPEG documents are allowed.")
    mime_type, signature = document_type
    return filename, mime_type, signature


def add_invoice_document(
    company_id: int,
    actor_user_id: int,
    invoice_id: int,
    original_filename: str,
    content: bytes,
) -> InvoiceDocument:
    filename, mime_type, signature = _safe_upload_name(original_filename)
    if not isinstance(content, bytes) or not content:
        raise ValueError("The document is empty or invalid.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("Documents must be 10 MB or smaller.")
    if not content.startswith(signature):
        raise ValueError("The document contents do not match the selected file type.")

    extension = ".jpg" if mime_type == "image/jpeg" else Path(filename).suffix.lower()
    storage_key = f"{uuid.uuid4().hex}{extension}"
    root = UPLOAD_ROOT.resolve()
    company_directory = (root / f"company-{company_id}").resolve()
    if root not in company_directory.parents:
        raise ValueError("Invalid document storage location.")
    company_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_path = company_directory / storage_key
    temporary_path = company_directory / f"{storage_key}.tmp"

    with SessionLocal() as session:
        _require_finance_user(session, company_id, actor_user_id)
        invoice = session.scalar(
            select(Invoice).where(
                Invoice.id == invoice_id,
                Invoice.company_id == company_id,
            )
        )
        if invoice is None:
            raise ValueError("Invoice was not found in this company.")
        if invoice.status != "pending":
            raise ValueError("Documents can only be attached to pending invoices.")

        try:
            with temporary_path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, target_path)
            document = InvoiceDocument(
                company_id=company_id,
                invoice_id=invoice.id,
                storage_key=storage_key,
                original_filename=filename,
                mime_type=mime_type,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                uploaded_by_user_id=actor_user_id,
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        except Exception:
            session.rollback()
            temporary_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            raise


def read_invoice_document(
    company_id: int,
    document_id: int,
) -> tuple[InvoiceDocument, bytes]:
    with SessionLocal() as session:
        document = session.scalar(
            select(InvoiceDocument)
            .join(
                Invoice,
                and_(
                    Invoice.id == InvoiceDocument.invoice_id,
                    Invoice.company_id == InvoiceDocument.company_id,
                ),
            )
            .where(
                InvoiceDocument.id == document_id,
                InvoiceDocument.company_id == company_id,
                Invoice.company_id == company_id,
            )
        )
        if document is None:
            raise ValueError("Document was not found in this company.")
        key_match = STORED_KEY_PATTERN.fullmatch(document.storage_key)
        if key_match is None:
            raise ValueError("Document storage reference is invalid.")
        path = (
            UPLOAD_ROOT.resolve()
            / f"company-{company_id}"
            / document.storage_key
        ).resolve()
        root = UPLOAD_ROOT.resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Document file is unavailable.")
        content = path.read_bytes()
        if not hmac.compare_digest(
            hashlib.sha256(content).hexdigest(), document.sha256
        ):
            raise ValueError("Document integrity check failed.")
        return document, content
def create_company_member(*args, **kwargs):
    pass
