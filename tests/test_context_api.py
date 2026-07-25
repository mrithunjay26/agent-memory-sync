import os
import tempfile
import unittest
from unittest import mock

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import embeddings
import store
from app import app, require_admin, require_auth


class ContextApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["AGENT_MEMORY_DB_PATH"] = os.path.join(self.tmpdir.name, "store.db")
        self.project = "/tracked/repo"
        store.register_project(self.project)
        app.dependency_overrides[require_auth] = lambda: "tester"
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()
        os.environ.pop("AGENT_MEMORY_DB_PATH", None)
        self.tmpdir.cleanup()

    def test_context_route_requires_authentication(self):
        app.dependency_overrides.clear()
        response = self.client.get("/api/context", params={"project": self.project})
        self.assertEqual(response.status_code, 401)

    def test_retrieval_status_reports_the_model_search_actually_uses(self):
        """The dashboard advertises hybrid retrieval, so this endpoint has to
        report the real state rather than the intended one."""
        response = self.client.get("/api/retrieval/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], embeddings.active_model())
        self.assertEqual(body["semantic"], embeddings.is_semantic())
        self.assertEqual(body["rrf_k"], store.RRF_K)
        self.assertEqual(body["min_similarity"], store.semantic_threshold())
        self.assertTrue(body["detail"])

    def test_retrieval_status_admits_when_the_embedding_model_is_unavailable(self):
        """Degrading to lexical hashing is acceptable. Reporting it as semantic
        search would not be."""
        with mock.patch.object(embeddings, "_load_model", return_value=None):
            body = self.client.get("/api/retrieval/status").json()

        self.assertFalse(body["semantic"])
        self.assertEqual(body["model"], embeddings.FALLBACK_MODEL_ID)
        self.assertEqual(body["min_similarity"], store.SEMANTIC_MIN_SIMILARITY)

    def test_retrieval_status_requires_authentication(self):
        app.dependency_overrides.clear()
        self.assertEqual(self.client.get("/api/retrieval/status").status_code, 401)

    def test_only_the_auth_endpoints_are_reachable_without_a_session(self):
        """The README promises every /api route is behind login. Adding a route
        and forgetting the dependency is a silent hole, so the route table is
        checked directly rather than one endpoint at a time.

        The /api/auth/* endpoints are the deliberate exception: requiring a
        session to log in would make logging in impossible.
        """
        allowed = {
            "/api/auth/status",
            "/api/auth/whoami",
            "/api/auth/signup",
            "/api/auth/login",
            "/api/auth/logout",
        }

        unprotected = set()
        for route in app.routes:
            if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
                continue
            guards = {dep.call for dep in route.dependant.dependencies}
            if not guards & {require_auth, require_admin}:
                unprotected.add(route.path)

        self.assertEqual(unprotected, allowed)

    def test_unknown_project_is_rejected(self):
        response = self.client.get("/api/context", params={"project": "/unknown"})
        self.assertEqual(response.status_code, 400)

    def test_update_rejects_event_from_another_project(self):
        store.record_event("/other", "codex", "s1", "turn", "private to other")
        event_id = store.get_context_bundle("/other")["entries"][0]["id"]
        response = self.client.patch(
            f"/api/context/events/{event_id}",
            params={"project": self.project},
            json={"included": False},
        )
        self.assertEqual(response.status_code, 404)

    def test_note_update_settings_and_delete_round_trip(self):
        created = self.client.post(
            "/api/context/notes",
            json={"project_path": self.project, "content": "Preserve the public API."},
        )
        self.assertEqual(created.status_code, 200)
        note = created.json()["entries"][0]

        updated = self.client.patch(
            f"/api/context/events/{note['id']}",
            params={"project": self.project},
            json={
                "pinned": True,
                "context_summary": "Keep the API stable.",
                "category": "constraint",
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertTrue(updated.json()["entries"][0]["pinned"])
        self.assertEqual(updated.json()["entries"][0]["summary"], "Preserve the public API.")
        self.assertEqual(updated.json()["entries"][0]["category"], "constraint")
        self.assertEqual(updated.json()["entries"][0]["category_source"], "manual")

        settings = self.client.put(
            "/api/context/settings",
            json={"project_path": self.project, "recent_limit": 7},
        )
        self.assertEqual(settings.status_code, 200)
        self.assertEqual(settings.json()["settings"]["recent_limit"], 7)

        deleted = self.client.delete(
            f"/api/context/notes/{note['id']}", params={"project": self.project}
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["entries"], [])

    def test_raw_event_cannot_be_deleted(self):
        store.record_event(self.project, "codex", "s1", "turn", "real event")
        event_id = store.get_context_bundle(self.project)["entries"][0]["id"]
        response = self.client.delete(
            f"/api/context/notes/{event_id}", params={"project": self.project}
        )
        self.assertEqual(response.status_code, 400)

    def test_note_and_limit_validation(self):
        empty = self.client.post(
            "/api/context/notes", json={"project_path": self.project, "content": "  "}
        )
        self.assertEqual(empty.status_code, 400)
        invalid_limit = self.client.put(
            "/api/context/settings",
            json={"project_path": self.project, "recent_limit": 101},
        )
        self.assertEqual(invalid_limit.status_code, 400)
        invalid_category = self.client.post(
            "/api/context/notes",
            json={"project_path": self.project, "content": "x", "category": "mystery"},
        )
        self.assertEqual(invalid_category.status_code, 400)


if __name__ == "__main__":
    unittest.main()
