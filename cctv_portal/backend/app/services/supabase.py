from supabase import create_client, Client
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class SupabaseService:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Use service role for backend
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        self.client: Client = create_client(self.url, self.key)

    async def log_violation(self, violation_data: dict):
        """
        Master logic to populate:
        1. cctv_events (Table 1)
        2. cctv_evidence (Table 2)
        """
        try:
            def make_serializable(obj):
                if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
                if isinstance(obj, (np.float64, np.float32)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list): return [make_serializable(i) for i in obj]
                return obj

            raw_ts = violation_data.get("timestamp")
            valid_ts = datetime.fromtimestamp(raw_ts).isoformat() if isinstance(raw_ts, (int, float)) else (raw_ts or datetime.now().isoformat())

            # 1. Insert into cctv_events (Table 1)
            event_id = violation_data.get("id") # Try to use Python ID
            event_payload = {
                "camera_id": str(violation_data.get("camera_id", "UNKNOWN")),
                "event_type": str(violation_data.get("type") or violation_data.get("violation_type", "unknown")),
                "severity": 1,
                "occurred_at": valid_ts,
                "review_status": "PENDING"
            }
            if event_id: event_payload["id"] = event_id # Attempt to force Python ID

            # Map severity
            s_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            event_payload["severity"] = s_map.get(str(violation_data.get("severity", "LOW")).upper(), 1)

            e_res = self.client.table("cctv_events").insert(event_payload).execute()
            if not e_res.data:
                raise Exception("Table 'cctv_events' insert returned no data")
            
            # AUTHORITATIVE ID from database
            true_event_id = e_res.data[0]['id']
            print(f"✅ TABLE 1 SYNCED: cctv_events (DB ID: {true_event_id})")

            # 2. Insert into cctv_evidence (Table 2)
            # Match schema columns carefully
            student_id = violation_data.get("student_id") or violation_data.get("candidate_id")
            ai_conf = violation_data.get("ai_confidence") or violation_data.get("confidence") or 0.95
            
            evidence_payload = {
                "cctv_event_id": true_event_id,
                "camera_id": event_payload["camera_id"],
                "student_id": str(student_id) if student_id else "UNKNOWN",
                "evidence_type": "VIDEO",
                "storage_url": violation_data.get("evidence_url"),
                "ai_confidence": float(ai_conf),
                "severity": event_payload["severity"],
                "captured_at": valid_ts,
                "detected_objects": str(violation_data.get("detected_objects", "")),
                "metadata": make_serializable(violation_data.get("metadata", {})),
                "duration_seconds": int(violation_data.get("duration_seconds", 0)),
                "file_size_bytes": int(violation_data.get("file_size_bytes", 0)),
                "hash_sha256": violation_data.get("hash_sha256"),
                "encoding_format": "mp4",
                "retained_until": violation_data.get("retained_until")
            }

            ev_res = self.client.table("cctv_evidence").insert(evidence_payload).execute()
            if not ev_res.data:
                print(f"❌ TABLE 2 INSERT FAILED for Event {true_event_id}")
            else:
                print(f"✅ TABLE 2 SYNCED: cctv_evidence")

            # Return the authoritative record for the UI
            return [{**event_payload, **evidence_payload, "id": true_event_id}]

        except Exception as e:
            print(f"❌ SUPABASE SYNC FATAL: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def log_malpractice_event(self, event_data: dict, audit_log_id: str):
        """
        Populate:
        1. malpractice_events (Table 4)
        2. evidence (Table 5 - Malpractice to Audit link)
        """
        try:
            # 1. malpractice_events (Table 4)
            m_payload = {
                "attempt_id": str(event_data.get("attempt_id", "UNKNOWN")),
                "event_type": str(event_data.get("event_type", "unknown")),
                "severity": int(event_data.get("severity", 1)),
                "source": "PROCTOR_CONFIRMED",
                "description": str(event_data.get("description", "Confirmed Cheating")),
                "occurred_at": event_data.get("occurred_at")
            }
            m_res = self.client.table("malpractice_events").insert(m_payload).execute()
            if not m_res.data:
                raise Exception("Table 'malpractice_events' insert failed")
            
            m_id = m_res.data[0]['id']
            print(f"✅ TABLE 4 SYNCED: malpractice_events (ID: {m_id})")

            # 2. evidence (Table 5 - Master linkage)
            e_payload = {
                "malpractice_id": m_id,
                "evidence_type": "VIDEO",
                "storage_url": event_data.get("evidence_url"),
                "audit_log_id": audit_log_id, # Linking confirmed case to specific proctor action
                "captured_at": event_data.get("occurred_at"),
                "hash_sha256": event_data.get("hash_sha256")
            }
            e_res = self.client.table("evidence").insert(e_payload).execute()
            if not e_res.data:
                print(f"❌ TABLE 5 INSERT FAILED for Malpractice {m_id}")
            else:
                print(f"✅ TABLE 5 SYNCED: evidence (link to audit)")

            return m_res.data
        except Exception as e:
            print(f"❌ MALPRACTICE LOG FATAL: {e}")
            return None

    async def upload_evidence(self, file_path: str, bucket_name: str = "cctv_evidence"):
        """Upload evidence file (JPEG or MP4) to Supabase Storage"""
        try:
            file_name = os.path.basename(file_path)
            content_type = "video/mp4" if file_name.endswith(".mp4") else "image/jpeg"
            
            with open(file_path, "rb") as f:
                self.client.storage.from_(bucket_name).upload(
                    file_name, f, {"content-type": content_type, "upsert": "true"}
                )
            
            public_url = self.client.storage.from_(bucket_name).get_public_url(file_name)
            return public_url
        except Exception as e:
            print(f"Supabase Upload Error: {e}")
            return None
