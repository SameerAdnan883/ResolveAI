import requests
from typing import Dict, Any, Optional, List
from backend.config import settings

class ResolveAIClient:
    def __init__(self, base_url: str = settings.BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def _headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def health(self) -> Dict[str, Any]:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=5)
            return res.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def login(self, email: str, password: str) -> Dict[str, Any]:
        res = requests.post(
            f"{self.base_url}/auth/login",
            json={"email": email, "password": password},
            timeout=10
        )
        return res.json() if res.status_code == 200 else {"error": res.json().get("detail", "Login failed")}

    def register(self, email: str, password: str, full_name: str, phone: str = "") -> Dict[str, Any]:
        res = requests.post(
            f"{self.base_url}/auth/register",
            json={"email": email, "password": password, "full_name": full_name, "phone": phone},
            timeout=10
        )
        return res.json() if res.status_code == 200 else {"error": res.json().get("detail", "Registration failed")}

    def submit_complaint(self, token: str, complaint_text: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        res = requests.post(
            f"{self.base_url}/tickets/submit",
            headers=self._headers(token),
            json={"complaint_text": complaint_text, "order_id": order_id},
            timeout=30
        )
        return res.json() if res.status_code == 200 else {"error": res.json().get("detail", "Submission failed")}

    def get_my_tickets(self, token: str) -> List[Dict[str, Any]]:
        res = requests.get(f"{self.base_url}/tickets/my", headers=self._headers(token), timeout=10)
        return res.json() if res.status_code == 200 else []

    def get_all_tickets(self, token: str, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/tickets/all"
        if status_filter:
            url += f"?status_filter={status_filter}"
        res = requests.get(url, headers=self._headers(token), timeout=10)
        return res.json() if res.status_code == 200 else []

    def get_ticket_details(self, token: str, ticket_id: str) -> Dict[str, Any]:
        res = requests.get(f"{self.base_url}/tickets/{ticket_id}", headers=self._headers(token), timeout=10)
        return res.json() if res.status_code == 200 else {"error": "Ticket details failed"}

    def get_pending_approvals(self, token: str) -> List[Dict[str, Any]]:
        res = requests.get(f"{self.base_url}/tickets/approvals/pending", headers=self._headers(token), timeout=10)
        return res.json() if res.status_code == 200 else []

    def review_approval(self, token: str, approval_id: int, approved: bool, admin_notes: Optional[str] = None) -> Dict[str, Any]:
        res = requests.post(
            f"{self.base_url}/tickets/approvals/{approval_id}/review",
            headers=self._headers(token),
            json={"approved": approved, "admin_notes": admin_notes},
            timeout=10
        )
        return res.json() if res.status_code == 200 else {"error": res.json().get("detail", "Review failed")}

    def get_analytics_summary(self, token: str) -> Dict[str, Any]:
        res = requests.get(f"{self.base_url}/analytics/summary", headers=self._headers(token), timeout=10)
        return res.json() if res.status_code == 200 else {}

